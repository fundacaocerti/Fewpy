# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 0.1.0 - XXXX/XX/XX
******
### Added
   - Utility 'FSLDataset'  [3](https://github.com/fundacaocerti/Fewpy/pull/3)
   - Utility 'FewShotModel' [5](https://github.com/fundacaocerti/Fewpy/pull/5)
   - Utility 'Preprocessor' [5](https://github.com/fundacaocerti/Fewpy/pull/5)
   - Model 'FPTRANS' [4](https://github.com/fundacaocerti/Fewpy/pull/4)
   - Model 'AnomalyCLIP' [5](https://github.com/fundacaocerti/Fewpy/pull/5)
   - Model 'AirShot' [6](https://github.com/fundacaocerti/Fewpy/pull/6)
   - Model 'Qwen' [7](https://github.com/fundacaocerti/Fewpy/pull/7)
   - Utility 'CVATAdapter' [9](https://github.com/fundacaocerti/Fewpy/pull/9)
   - Documentation 'Fewpy/examples' [9](https://github.com/fundacaocerti/Fewpy/pull/9)
   - Model 'TipAdapter' [20](https://github.com/fundacaocerti/Fewpy/pull/20)

### Changed
   - Utility 'FSLDataset' now supports shortest edge resizing [6](https://github.com/fundacaocerti/Fewpy/pull/6)
   - Utility 'FSLDataset' now supports bounding box normalization [7](https://github.com/fundacaocerti/Fewpy/pull/7)
   - Model 'FPTRANS' had outputs adapted to the libraries new standard [8](https://github.com/fundacaocerti/Fewpy/pull/8)
   - Model 'AnomalyCLIP' had outputs adapted to the libraries new standard [8](https://github.com/fundacaocerti/Fewpy/pull/8)
   - Model 'AirShot' had outputs adapted to the libraries new standard [8](https://github.com/fundacaocerti/Fewpy/pull/8)
   - Model 'Qwen' had outputs adapted to the libraries new standard [8](https://github.com/fundacaocerti/Fewpy/pull/8)
   - Model 'Airshot' had its weight path search logic improved [9](https://github.com/fundacaocerti/Fewpy/pull/9)
   - Model 'AnomalyCLIP' had its weight path search logic improved [9](https://github.com/fundacaocerti/Fewpy/pull/9)
   - Utility 'FSLDataset' is now used directly with torch.utils.data.DataLoader [11](https://github.com/fundacaocerti/Fewpy/pull/11)
   - Documentation 'Fewpy/examples' cvat exmple import typo fixed [11](https://github.com/fundacaocerti/Fewpy/pull/11)
   - Model 'Airshot' now supports batched inputs (query) [11](https://github.com/fundacaocerti/Fewpy/pull/11)
   - Model 'Qwen' now supports batched inputs (query) [11](https://github.com/fundacaocerti/Fewpy/pull/11)
   - Model 'AnomalyCLIP' fixed prompt learner weight loading isses [11](https://github.com/fundacaocerti/Fewpy/pull/11)
   - Utility 'FSLDataset' now implements multiple preprocessing routines to be used during inference and fine tuning [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Utility 'FewShotModel' improved to support model fine tuning [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Model 'AnomalyCLIP' predict method updated to support training mode [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Model 'Airshot' predict method updated to support training mode [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Model 'Qwen' predict method updated to support training mode [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Model 'Qwen' supports LoRA and Quantization [17](https://github.com/fundacaocerti/Fewpy/pull/17)
   - Utility 'FSLDataset' fixed multiprocessing bug when preprocessing support set [20](https://github.com/fundacaocerti/Fewpy/pull/20)
   - Utility 'FSLDataset' added new preprocessing method to be used with TipAdapter [20](https://github.com/fundacaocerti/Fewpy/pull/20)
   - Utility 'FSLDataset' uses PreprocessingMethod(Enum) for preprocessing method selection [20](https://github.com/fundacaocerti/Fewpy/pull/20)
   - Utility 'FSLDataset' fixed bounding box normalization bug [20](https://github.com/fundacaocerti/Fewpy/pull/20)
   - Model 'FPTRANS' predict updated to support training mode (test pending) [20](https://github.com/fundacaocerti/Fewpy/pull/20)
