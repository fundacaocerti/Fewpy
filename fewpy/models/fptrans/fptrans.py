from collections import OrderedDict
from functools import partial
from fewpy.util.inference.register import register_constructor
from .config import FPTRANSConfig
#from losses import get as get_loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

from .base.vit import vit_model, vit_factory
from pathlib import Path


interpb = partial(F.interpolate, mode='bilinear', align_corners=True)
interpn = partial(F.interpolate, mode='nearest')

class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x


class FPTRANS(nn.Module):
    def __init__(self, args, logger):
        super(FPTRANS, self).__init__()
        
        self.Probs_return = args.Probs_return
        # self.logger = logger
        self.args = args
        self.shot = args.kshot
        self.drop_dim = args.drop_dim
        self.drop_rate = args.drop_rate
        self.drop2d_kwargs = {'drop_prob': args.drop_rate, 'block_size': args.block_size}

        # Check existence.
        pretrained = args.pretrained

        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit_model(args.backbone,
                                       args.height,
                                       pretrained=pretrained,
                                       num_classes=0,
                                       args=args))
        ]))
        embed_dim = vit_factory[args.backbone]['embed_dim']
        self.purifier = self.build_upsampler(embed_dim)
        self.__class__.__name__ = f"FPTrans/{args.backbone}"

        # Pretrained model
        self.original_encoder = vit_model(args.backbone,
                                        args.height,
                                        pretrained=pretrained,
                                        num_classes=0,
                                        args=args,
                                        original=True)
        for var in self.original_encoder.parameters():
            var.requires_grad = False

        # Define pair-wise loss
        #self.pairwise_loss = get_loss(args, logger, loss='pairwise')

        # Background sampler
        self.bg_sampler = np.random.RandomState(1289)

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))

    def forward(self, x, s_x, s_y, y=None, out_shape=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            [B, C, H, W], query image
        s_x: torch.Tensor
            [B, S, C, H, W], support image
        s_y: torch.Tensor
            [B, S, H, W], support mask
        y: torch.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [cfg.train.height, cfg.train.width].

        Returns
        -------
        output: dict
            'out': torch.Tensor
                logits that predicted by feature proxies
            'out_prompt': torch.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """


        B, S, C, H, W = s_x.size()
        x0 = x.clone()
        x1 = torch.cat((s_x, x0.view(B, 1, C, H, W)), dim=1)
        x2 = x1.clone()
        x2 = x2.view(B*(S+1), C, H, W)

        # Calculate class-aware prompts
        with torch.no_grad():
            s_x0 = s_x.clone()
            inp = s_x0.view(B * S, C, H, W)
            # Forward
            sup_feat = self.original_encoder(inp)['out']
            _, c, h0, w0 = sup_feat.shape
            s_y0 = s_y.clone()
            sup_mask = interpn(s_y0.view(B*S, 1, H, W), (h0, w0))                                # [BS, 1, h0, w0]
            sup_mask_fg = (sup_mask == 1).float()                                               # [BS, 1, h0, w0]
            # Calculate fg and bg tokens
            fg_token = (sup_feat * sup_mask_fg).sum((2, 3)) / (sup_mask_fg.sum((2, 3)) + 1e-6)
            fg_token0 = fg_token.clone()
            fg_token0 = fg_token0.view(B, S, c).mean(1, keepdim=True) # [B, 1, c]
            sup_feat0 = sup_feat.clone()
            bg_token = self.compute_multiple_prototypes(
                self.args.bg_num,
                sup_feat0.view(B, S, c, h0, w0),
                sup_mask == 0,
                self.bg_sampler
            )
            bg_token0 = bg_token.clone()
            bg_token0 = bg_token0.transpose(1, 2)    # [B, k, c]

        # Forward
        img_cat = (x2, (fg_token0, bg_token0))
        backbone_out = self.encoder(img_cat)

        features0 = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features0.size()

        features = features0.clone()
        features = features.view(B, S+1, c, h, w)                 # [B, S+1, c, h, w]
        sup_fts, qry_fts = features.split([S, 1], dim=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        s_y1 = s_y.clone()
        sup_mask0 = interpn(s_y1.view(B * S, 1, H, W), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask0)          # [B, 2, h, w]
        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        #output = dict(out=out)
        
        """ 
        if self.args.training and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                # [B, S, C, N]
            y1 = sup_mask0.view(B, S, -1).long()     # [B, S, N] # maybe need clone
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]

            y2 = interpn(y.float(), (h, w)).flatten(2).long()   # [B, 1, N]

            output['loss_pair'] = self.pairwise_loss(x1, y1, x2, y2)

            # Prompt-Proxy prediction
            fg_token = self.purifier(backbone_out['tokens']['fg'])[:, :, 0, 0]        # [B, c]
            bg_token = self.purifier(backbone_out['tokens']['bg'])[:, :, 0, 0]        # [B, c]
            bg_token = bg_token.view(B, self.args.bg_num, c).transpose(1, 2)     # [B, c, k]
            pred_prompt = self.compute_similarity(fg_token, bg_token, qry_fts.reshape(-1, c, h, w))

            # Up-sampling
            pred_prompt = interpb(pred_prompt, (H, W))
            output['out_prompt'] = pred_prompt

            return output
            
        else:
        """
        if self.Probs_return:
            return out #[bsz, 2, H, W] 
        else:
            return out.argmax(dim=1) #[bsz, H, W] 
        

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w) # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1).clone()    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(self.args.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
            [B, S, c, h, w], float32
        sup_bg: torch.Tensor
            [BS, 1, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        sup_bg0 = sup_bg.clone()
        bg_mask = sup_bg0.view(B, S, h, w)   # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with torch.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.view(-1)[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = torch.stack(torch.where(bg_mask_i), dim=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                        # choose the farthest point
                        i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)
            
                # Assign bg labels for bg pixels
                dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].permute(1, 2, 0)[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [B, 2, h, w]

        return pred
    '''
    def load_weights(self, ckpt_path, device):
        """

        Parameters
        ----------
        ckpt_path: Path
            path to the checkpoint
        strict: bool
            strict mode or not

        """
        weights = torch.load(ckpt_path, map_location=device)
        if "model_state" in weights:
            weights = weights["model_state"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})
        self.load_state_dict(weights) 
    '''       

    ''' 
    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file
    '''

    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]

    def predict(self, batch):
        
        if self.args.SAHI:
            
            line, column = batch['query_img'].size()[1:3]
            pred_mask_batch = torch.zeros(self.args.bsz, line, column, 2, self.args.img_size, self.args.img_size)

            for i in range(line):
                for j in range(column):
                    pred_mask_batch[:,i,j,:,:,:]  = self(batch['query_img'][:,i,j,:,:,:], batch['support_imgs'],batch['support_masks'])

            list_column = [pred_mask_batch[:, :, i, :, :, :] for i in range(column)]
            cat_list = list(map(lambda x: x.squeeze(2), list_column))

            # Concatenar com as imagens da primeira coluna ao longo da nova dimensão
            novo_tensor = torch.cat(cat_list, dim=4)

            list_column = [novo_tensor[:,i,:, :, :] for i in range(line)]
            cat_list = list(map(lambda x: x.squeeze(1), list_column))
            novo_tensor = torch.cat(cat_list, dim=2)

            return novo_tensor

        return self(batch['query_img'], batch['support_imgs'],batch['support_masks'])


@register_constructor(name="FPTRANS", config_cls=FPTRANSConfig)
class constructor_FPTRANS():
    def __init__(self, args):
            
        self.args = args

    def construct_yaml(self):

        path_yaml = Path("./weights") / f"{self.args.data_set}" \
            / f"{self.args.data_set}_{self.args.backbone}_fold{self.args.split}_{self.args.kshot}shot.yaml"

        return path_yaml

    def instantiate_model(self):

        model = FPTRANS(self.args, None)
        return self.load_weights(model)

    def load_weights(self, model):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if not self.args.checkpoint is None:
            state_dict = self.args.checkpoint
        else:
            weight_path = Path("./weights").expanduser() / f"{self.args.data_set}" \
                / f"{self.args.data_set}_{self.args.backbone}_fold{self.args.split}_{self.args.kshot}shot.pth"
            weights = torch.load(weight_path, map_location=device)

            if "model_state" in weights:
                state_dict = weights["model_state"]
            if "state_dict" in weights:
                state_dict = weights["state_dict"]

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict.update({k: v for k, v in model.state_dict().items() if 'original_encoder' in k})

        model.load_state_dict(state_dict)        

        if self.args.training:
        
            return model.train()

        return model.eval(), device
