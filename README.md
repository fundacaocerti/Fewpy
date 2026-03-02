# Fewpy

An Open Source project for Few-Shot Learning (FSL). 

Fewpy implements state-of-art Image Segmentation FSL models. It allows the user to easily load and use the following models for inference:

## Use cases:

### Proof of Concept

Fewpy can be used to verify feasability of tasks in SOTA FSL models and support proof of concept.

### Dataset Annotation

Fewpy enables faster data annotation by simplifying the use of cutting-edge AI models.

### Benchmarking

Fewpy makes it easy to load and run highly performant models on benchmarks to experiment or compare with other model's results.

## Models

### AnomalyCLIP

### FPTRANS

### AirShot (FSOD RCNN)

Copyright (c) 2024, Zihan Wang

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This model is an implementation of the efficiency improvements proposed by [AirShot: Efficient Few-Shot Detection for Autonomous Exploration](https://arxiv.org/abs/2404.05069) (Wang et al., 2024). It consists in an FSOD RCNN built with detectron2. Thus it is a few-shot detection model.

For more detailed information on expected input or configuration refer to [./examples/models/airshot.md]()
