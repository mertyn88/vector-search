# Similar vector search in Elasticsearch

Elasticsearch를 통해 유사한 문서를 검색하는 예제  

* Elasticsearch version : 8.5.2
* DataSet : [`한국 관광지`](https://www.data.go.kr/data/15021141/standard.do)
* Python : 3.10


## Docker

### [docker-compose.yml](./docker-compose.yml)
**Important note**: Localhost ES, Kibana 8.5.2를 사용  
* Elasticsearch Node 2
* Kibana

### run docker
```bash
docker-compose -f ./docker-compose.yml up -d
```

## Jupyter
해당 예제는 jupyter-notebook을 통해서만 테스트가 진행

### Code

```python
# Set Import
from elasticsearch import Elasticsearch
import pandas
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pynori.korean_analyzer import KoreanAnalyzer
```


```python
# Set Nori Analyzer
nori = KoreanAnalyzer(
           decompound_mode='DISCARD', # DISCARD or MIXED or NONE
           infl_decompound_mode='DISCARD', # DISCARD or MIXED or NONE
           discard_punctuation=True,
           output_unknown_unigrams=False,
           synonym_filter=False, mode_synonym='NORM', # NORM or EXTENSION
)

# NNG, NNP (명사, 대명사) filter
def _filter(term):
    result = []
    for _idx, _tag in enumerate(term['posTagAtt']):
        if _tag in ['NNG', 'NNP']:
            result.append(term['termAtt'][_idx])
    return result

# Analyzer
def _do_analysis(text):
    return _filter(nori.do_analysis(text))

# 띄어쓰기 Tokenizer
def _nltk_tokenizer(_wd):
  return RegexpTokenizer(r'\w+').tokenize(_wd.lower())

print(_do_analysis("아빠가 방에 들어가신다."))
```

    ['아빠', '방']



```python
# Set frame
frame = pandas.read_csv('../data/travel.csv', encoding='utf-8').fillna('')
frame = frame.query("description != ''")
frame = frame.reset_index(drop=False)
frame['token_description'] = frame['description'].apply(_do_analysis)
# Token length > 0 filter
frame = frame.query("token_description.str.len() > 0")
data = {
    'seq': frame['index'],
    'title': frame['title'].tolist(),
    'description': frame['description'].tolist(),
    'token_description': frame['token_description'].tolist()
}
frame = pandas.DataFrame(data)
frame.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seq</th>
      <th>title</th>
      <th>description</th>
      <th>token_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>어답산관광지</td>
      <td>병지방계곡,캠핑장</td>
      <td>[병지방, 계곡, 캠, 핑장]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>유현문화관광지</td>
      <td>풍수원성당, 유물전시관, 산책로</td>
      <td>[풍, 수원, 성당, 유물, 전시, 관, 산책, 로]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>웰리힐리파크 관광단지</td>
      <td>스키장, 골프장, 곤돌라, 콘도</td>
      <td>[스키, 장, 골프, 장, 곤돌라, 콘]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set TaggedDocument, [seq, token]
frame_doc = frame[['seq','token_description']].values.tolist()
tagged_data = [TaggedDocument(words=_d, tags=[uid]) for uid, _d in frame_doc]
tagged_data[:3]
```




    [TaggedDocument(words=['병지방', '계곡', '캠', '핑장'], tags=[0]),
     TaggedDocument(words=['풍', '수원', '성당', '유물', '전시', '관', '산책', '로'], tags=[1]),
     TaggedDocument(words=['스키', '장', '골프', '장', '곤돌라', '콘'], tags=[2])]




```python
# Train
model = Doc2Vec(
    window=3,          # window: 모델 학습할때 앞뒤로 보는 단어의 수
    vector_size=100,    # size: 벡터 차원의 크기
    alpha=0.025,        # alpha: learning rate
    min_alpha=0.025,
    min_count=2,        # min_count: 학습에 사용할 최소 단어 빈도 수
    dm = 0,              # dm: 학습방법 1 = PV-DM, 0 = PV-DBOW
    negative = 5,       # negative: Complexity Reduction 방법, negative sampling
    seed = 9999
)

"""
epoch
한 번의 epoch는 인공 신경망에서 전체 데이터 셋에 대해 forward pass/backward pass 과정을 거친 것을 말함. 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태

batch size
batch size는 한 번의 batch마다 주는 데이터 샘플의 size. 여기서 batch(보통 mini-batch라고 표현)는 나눠진 데이터 셋을 뜻하며 iteration는 epoch를 나누어서 실행하는 횟수
메모리의 한계와 속도 저하 때문에 대부분의 경우에는 한 번의 epoch에서 모든 데이터를 한꺼번에 집어넣을 수는 없습니다. 그래서 데이터를 나누어서 주게 되는데 이때 몇 번 나누어서 주는가를 iteration, 각 iteration마다 주는 데이터 사이즈를 batch size
"""

max_epochs = 10
model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

# store the model to mmap-able files
#model.save('./model.doc2vec')
# load the model back
#model_loaded = Doc2Vec.load('./model.doc2vec')

model.wv.vectors[:3]
```
    array([[-8.2193622e-03,  6.7749298e-03, -6.2365090e-03,  5.2402792e-03,
            -4.6296441e-03, -4.7527459e-03,  2.4112677e-03,  5.7777343e-03,
            -1.2703657e-03, -5.1266574e-03, -1.7415201e-03,  1.5200340e-03,
             7.4816141e-03,  1.1664641e-03, -5.5718194e-03,  7.0228814e-03,
            -1.7327452e-03, -8.0134859e-03,  2.1830511e-03,  6.5284539e-03,
             2.0196950e-03,  1.3344670e-03, -8.6302888e-03, -1.4463830e-03,
            -3.6384177e-03, -1.4376461e-03,  4.2982958e-03, -5.0189043e-03,
            -3.2979273e-03,  6.0618282e-03,  2.3644101e-03, -6.6062331e-04,
            -8.4960023e-03,  8.7066712e-03,  4.5560561e-03, -4.6418346e-03,
             7.5713051e-03, -7.0132697e-03, -6.5153148e-03,  3.6592449e-03,
             7.5526941e-03, -9.8076165e-03,  5.4272129e-03, -8.2187047e-03,
             7.0318556e-03,  5.1973341e-03,  1.5876698e-03, -5.4299831e-03,
             7.5712288e-03, -9.9267438e-03,  8.5650627e-03,  1.1790371e-03,
            -3.8849604e-03,  6.8626092e-03,  1.1697662e-03,  8.6435378e-03,
            -1.1592471e-03, -4.3920744e-03,  2.1206522e-03,  4.7955285e-03,
            -4.0982963e-05,  7.9155937e-03, -9.3902657e-03, -1.3651741e-03,
            -3.9263366e-04, -5.0132619e-03, -4.3503047e-04, -2.4629128e-03,
             9.6908044e-03,  6.9340039e-03, -2.8083325e-05,  4.9437879e-04,
             8.5202614e-03, -9.0881996e-03,  1.9592082e-03,  9.3845185e-03,
             7.0294854e-04, -9.8714354e-03,  2.4092877e-03, -8.9288400e-03,
            -5.4936018e-03, -7.2597242e-03,  6.5645743e-03, -5.0571309e-03,
             5.6476868e-03,  9.0139750e-03, -1.3698959e-03, -3.0222666e-03,
             1.9279898e-03,  3.3574998e-03,  6.1958479e-03,  9.8291012e-03,
             1.3717902e-03,  9.9756559e-03,  8.9205634e-03, -4.3565761e-03,
            -4.9003270e-03, -2.8724326e-03,  1.0620725e-03, -3.0619490e-03]
           ],
          dtype=float32)




```python
# Index docs
docs = [
    {
        '_index': 'vector_sample',
        '_source': {
            'title': _row['title'],
            'description': _row['description'],
            'description_vector': model.dv.vectors[_idx,:].tolist()
        }
    }
    for _idx, _row in frame.iterrows()
]
docs[0]
```
```json
{
    '_index': 'vector_sample',
    '_source': {
    'title': '어답산관광지',
    'description': '병지방계곡,캠핑장',
    'description_vector': 
        [
        -0.0748947337269783,
        -0.22988615930080414,
        -0.08189909905195236,
        0.17290998995304108,
        0.1827821135520935,
        0.12810979783535004,
        -0.07078631967306137,
        ...
        -0.19546788930892944,
        -0.007501175627112389,
        0.12139209359884262,
        0.10605379194021225,
        -0.20055268704891205,
        -0.05768788978457451,
        0.34618064761161804,
        -0.22516191005706787,
        0.0590231791138649,
        0.1486528366804123,
        0.14388953149318695,
        0.26719143986701965,
        0.17399340867996216,
        0.16179141402244568,
        0.35135209560394287,
        -0.23531629145145416
        ]
    }
}
```



```python
# Set Elasticsearch client ( == 8.5.2 )
client = Elasticsearch(hosts='http://127.0.0.1:9200')
```


```python
# Create Sample Index
client.indices.create(index='vector_sample', body={
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
   "mappings": {
    "properties": {
      "title": {
        "type": "keyword"
      },
      "description": {
        "type": "keyword"
      },
      "description_vector": {
        "type": "dense_vector",
        "dims": 100
      }
    }
  }
}
)
```



    ObjectApiResponse({'acknowledged': True, 'shards_acknowledged': True, 'index': 'vector_sample'})




```python
# Index bulk
from elasticsearch import helpers
res = helpers.bulk(client, docs)
res
```




    (721, [])




```python
# Search Sample
test = model.infer_vector(_do_analysis('옛날 스무나무 아래 약수가 있어 이를 마시고 위장병과 피부병에 효험이 있어 많은 사람이 이 약수를 마시고 덕을 보았다 하여 다덕약수라고 불리움'))
test.tolist()[:3]
```




    [-0.14725027978420258, -0.20942197740077972, -0.06804680079221725]




```python
# Query
script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'description_vector') + 1.0",
                "params": {"query_vector": test.tolist()}
            }
        }
    }
response = client.search(
        index="vector_sample",
        query=script_query,
        size=10,
        source_includes=["title", "description"]
    )
response['hits']['total']
```




    {'value': 721, 'relation': 'eq'}




```python
# Result
for idx, hit in enumerate(response["hits"]["hits"]):
    print('[' + str(idx) + ':' + str(hit["_score"]) + '] '  + hit["_source"]['description'])
```

    [0:1.9649781] 옛날 스무나무 아래 약수가 있어 이를 마시고 위장병과 피부병에 효험이 있어 많은 사람이 이 약수를 마시고 덕을 보았다 하여 다덕약수라고 불리움
    [1:1.9122046] 심산계곡에 자리잡은 약수탕은 선달산, 옥석산 아래 깊은 계곡에 위치하고 있고, 약수는 예부터 위장병과 피부병에 효험이 있다.
    [2:1.786823] 무등산을 느낄 수 있음
    [3:1.7127932] 데미샘은 3개도 10개 시군에 걸쳐 218.6㎞를 흐르는 우리나라에서 4번째로 긴강인 섬진강의 발원지이다
    [4:1.7120755] 중탄산 온천수 및 알칼리성 온천수 등 신진대사를 촉진하는 2가지 온천수가 있음
    [5:1.7029405] 숲과 계곡이 아름다운 청정도량
    [6:1.7003778] 옛날 석기 시대의 사람들이 이곳에서 살았으리라 짐작되는 혈거동굴로서 연구 가치가 매우 높다. 허준은 허가바위에서 『동의보감』을 완성했다고 한다.
    [7:1.6945602] 등명해변관광지
    [8:1.6934646] 온천수  - 수질 : 26.5℃ / PH 9.7(국내 최고의 강 알칼리성 수질)      ▶ 류머티즘, 알레르기성 피부염 등에 탁월한 효과
    [9:1.6844268] 온천과 약찜의 효능을 한꺼번에 즐길수 있음



```python

```
