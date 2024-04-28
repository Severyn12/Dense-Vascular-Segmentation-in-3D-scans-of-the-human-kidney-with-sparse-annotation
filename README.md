# Dense Vascular Segmentation in 3D scans of the human kidney with sparse annotation

All code used for the bachelor thesis can be found here
***

## Absract
The main goal of this study is to investigate how various loss functions affect the problem of sparse segmentation, based on the task of kidney vasculature segmentation. To achieve this we will conduct a comprehensive performance analysis of different losses, examining both their weak and strong sides. The evaluated loss functions include Binary Cross-Entrophy, its modification incorporating the Dice similarity coefficient(BCEDice), Focal loss, and Boundary Difference over the Union(BDoU) loss. Furthermore, this study will propose a novel loss criterion, which is a modified version of the standard BDoU loss(BDoUV2), incorporating an additional penalty for False Positive(FP) instances.

Experiments showed that loss functions exhibit different performances while encountering sparse segmentation issues, in particular, the poorest performance was achieved by the Focal loss, while the BDoUV2 loss along with the BCEDice loss showed the most prominent results. One common issue across these losses, to varying degrees, is that utilizing each of them results in a huge amount of FN errors. Moreover, we have explored, that all loss functions, except BCEDice loss, are highly affected by the value of the classification threshold, with the Focal loss being the most sensitive to it.

The insights regarding the performance of different loss functions under the task of sparse segmentation will be valuable for other scientists while developing an accurate and robust method for the task of medical sparse segmentation, where precision is critical for effective diagnosis and treatment planning.
***

## File navigation:
In this repository, you can find two different folders: Train and Inference, where the first one contains PyTorch scripts necessary for model training on GPU while the latter encompasses an inference notebook utilized for Kaggle competition submissions, along with another notebook featuring evaluation plots. Train scripts were taken from this [repository](https://github.com/burnmyletters/blood-vessel-segmentation-public) (P. S. Great thanks to BurnmyLetters for posting this code).

Saved model weights (use the last saved checkpoint e.g. ```last.ckpt```) along with tensorboard logs are placed on the Google Disk and can be found by this link

## Run Configuration:
 * Initially you should download the [Kaggle competition](https://www.kaggle.com/competitions/blood-vessel-segmentation/overview) dataset. This can be done through this [page](https://www.kaggle.com/competitions/blood-vessel-segmentation/data) or by running this command.

```bash
kaggle competitions download -c blood-vessel-segmentation
```

* Set up the environment using the commands below.

```bash
# clone project
git clone https://github.com/Severyn12/SenNet-HOA---Hacking-the-Human-Vasculature-in-3D.git

# navigate to the target directory
cd train/blood-vessel-segmentation-public

# [OPTIONAL] create a conda environment
conda create -n bvs python=3.10.9
conda activate bvs

# install the necessary project's requirements
pip install -r requirements.txt
```

* To train the model run the next command (by default will be used BDoU Loss).

```bash
sh ./train.sh
```

* To change the selected loss function you should modify ```configs/model/model.yaml``` file by uncommenting the necessary loss function

```bash
loss:
#  _target_: segmentation_models_pytorch.torch.nn.BCEWithLogitsLoss
#  _target_: segmentation_models_pytorch.losses.FocalLoss
#  _target_: src.models.components.losses.BCEDiceLoss
#  _target_: src.models.components.losses.BoundaryDoULossV2
_target_: src.models.components.losses.BoundaryDoULoss
```
