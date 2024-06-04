# Vision Transformer from Scratch 👶
기간 : 2023년 9월 ~ 11월 <br>
참고 문헌 : [An Image Is Worth 16 X 16 Words:Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929) <br>
해당 프로젝트는 Transformer 모델을 빠르고 간단하게 이해하고, 구현하는 것을 목적으로 진행하였습니다. <Br>

## Vision Transformer
Vision Transformer는 이미지 분류를 위해 Transformer 아키텍처를 적용한 모델로, 크게 두 단계로 나뉜다. <br>

| Input Data 생성 과정 | Transformer Encoder 과정 | 
|:---:|:---:|
|1. 이미지 분할<Br>(Image Patching) | 1. 멀티헤드 셀프 어텐션<br>(Multi-Head Self-Attention) |
|2. 패치 인코딩<br>(Patch Embedding) | 2. 피드포워드 네트워크<br>(Feedforwaed Network) |
|3. 위치 임베딩<br>(Position Embedding) | 3. 레이어 정규화 및 잔차 연결<br>(Layer Normalization and Residual Connections) |
|4. 클래스 토큰 추가<Br>(Class token) | |



### 1. Iunput Data 생성과정

![Input](https://github.com/kingodjerry/vision_transformer/assets/143167244/d345a51c-e3a9-4ee8-a3be-4608e672a24b)

해당 단계에서는 이미지를 Transformer가 처리할 수 있는 형태로 변환하는 과정이다. <br>
<br> 
**1. 이미지 분할(Image Patching)** <br>
이미지를 여러 개의 작은(Patch)로 분할한다. 224X224 크기의 이미지를 16X16 크기의 패치로 나누면 총 196개의 패치가 생성된다. <br>
**2. 패치 인코딩(Patch Embedding)** <br> 
각 패치를 1차원 백터로 변환한다. 패치를 펼쳐서 1차원 벡터로 만들고, 선형 변환을 적용하여 고정된 크기의 임베딩 벡터로 변환하는 과정이다. 16X16X3(RGB) 패치는 768차원의 벡터로 변환된다. <br>
**3. 위치 임베딩(Position Embedding)** <br> 
순서 정보를 유지하기 위해 각 패치 임베딩에 정보를 더해준다. 이는 Transformer의 self-attention 매커니즘이 패치 간의 상대적 위치를 인식할 수 있게 하기 위함이다. <br>
**4. 클래스 토큰(Class token)** <br> 
분류를 위해 특별한 클래스 토큰을 추가한다. 클래스 토큰은 학습과정에서 이미지 전체의 정보를 요약하는 역할을 한다. 이 토큰은 다른 패치 임베딩과 함께 Transformer에 입력된다. <br> 

### 2. Transformer Encoder 과정

![Encoder](https://github.com/kingodjerry/vision_transformer/assets/143167244/86bb3dae-5d37-4750-9bc5-735b10e7ff8f)

해당 단계에서는 앞 과정에서 생성된 입력 데이터를 Transformer Encoder를 통해 처리하는 과정이다. <br>
<br>
**1. 멀티헤드 셀프 어텐션(Multi-Head Self-Attention)** <br>
각 패치와 클래스 토큰의 관계를 학습한다. <br>
**2. 피드포워드 네트워크(Feedforward Network)** <br> 
어텐션 결과를 더 깊이 학습하기 위해 각 토큰 임베딩에 대해 독립적으로 적용되는 두 개의 완전 연결층을 포함한다. 일반적으로 활성화 함수로 GELU가 사용된다.  <br>
**3. 레이어 정규화 및 잔차 연결(Layer Normalization and Residual Connections)** <br> 
각 멀티헤드 셀프 어텐션과 피드포워드 네트워크 모듈의 출력에는 레이어 정규화와 잔차 연결이 적용된다. 이 과정은 학습을 안정화하고, 정보 흐름을 원활하게 유지할 수 있도록 한다. <br>

### 3. 최종 출력
Transformer Encoder의 마지막 출력에서 **클래스 토큰**을 추출하여, 최종적으로 분류 헤드(주로 선형 분류기)를 통해 이미지의 클래스를 예측한다. <br>


# ViT Fine-Tuning 🌾
앞서 이해한 Vision Transformer 모델을 Fine-tuning하여 **건강한 콩잎**과 **해로운 콩잎**으로 이미지를 분류하는 프로젝트를 진행하였다. 

### Pre-train model
Pre-train model : vit-base-patch-224-in21k model (구글 제공)

### Dataset
Dataset은 datasets transformers 라이브러리에서 'beans' 데이터를 사용하였다. <br>
해당 데이터셋은 콩잎 이미지 데이터로, 병에 걸린 콩잎과 건강한 콩잎으로 구분되어 있는 데이터 세트이다. <br>
Train : 1034, Validation : 133, Test : 128개로 구성되어 있으며, Label은 'angular_leaf_spot', 'bean_rust', 'healthy'로 구분되어 있다. <br> 

**1. transformer와 dataset 다운로드**
```
   pip install datasets transformers
   pip install transformers[torch]

   from datasets import load_dataset

   dataset = load_dataset('beans')
```
**2. dataset 확인** (각 클래스의 예제) <br>
**3. ViT 이미지 프로세서* Load** <br>
  *이미지 프로세서 : 모델이 이미지를 처리할 수 있도록 전처리하는 역할 수행(이미지 크기 조정, 패치 생성, 정규화, 텐서 변환 등등) <br>
**4. Input data로 변환** - tensor로 변환 <br>
**5. Pretrain model Load** <br>
```
from transformers import ViTForImageClassification

labels = ds['train'].features['labels'].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
```
**6. 파라미터 정의**
```
from transformers import ViTForImageClassification

labels = ds['train'].features['labels'].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
```
**7. 학습 준비**
```
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=processor,
)
```
**8. 학습**
```
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```
**9. 학습 평가**
```
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```
   

## Reference
```vision_transformer.ipynb```의 코드는 [tintn님의 vision-transformer-from-scratch](https://github.com/tintn/vision-transformer-from-scratch)에서 clone하였습니다. <br>
[참고 블로그 1 - Implementing Vision Transformer (ViT) from Scratch / Tin Nguyen ](https://towardsdatascience.com/implementing-vision-transformer-vit-from-scratch-3e192c6155f0) <br>
[참고 블로그 2 - [AI/ViT] Vision Transformer(ViT), 그림으로 쉽게 이해하기 / 미슈니 ](https://mishuni.tistory.com/137) <br>
