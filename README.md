음향 이벤트 구간 검출 성능 향상을 위해 11개월간 딥러닝 실험 연구를 진행하였습니다.(dcase2020 베이스라인 코드 사용)

Temporal Self-Attention and Guided Learning for Sound Event Detection
Toward advanced sound event detection model based on integration of feature pyramid and temporal attention(초록만 작성하여 발표) 

대한전자공학회 학술대회에서 두 논문 모두 각각 포스터 발표, 구두 발표로 연구내용 공유하였습니다.

베이스라인 모델 대비 성능 소폭 향상, 직접 모델 수정, 구축, 하이퍼파라미터들을 변화 시키며 성능을 최대로 끌어올리기 위한 연구를 하였습니다.

결과 요약

첫 번째 학술 논문
 
cat, speech, vacuum, frying 소리를 감지하는 부분에서 f1 score가 소폭 증가(+1.4%)했으며, electric 감지는 대폭 증가하였습니다. 반면, alarm bell, blender, dog, dishes 소리 감지 부분에서 f1 score가 소폭 감소하였습니다. 시간적 패턴이나 주파수 특성이 비교적 뚜렷하고 일관적인 소리들은 자가집중 모듈이 이를 학습하여 중요한 부분에 집중하였기 때문에 성능향상이 이루어진 반면, 발생 시점과 주파수 대역이 다양한 소리들은 자가집중 모듈이 학습하기 어려워 성능이 낮아졌습니다.

두 번째 학술논문

다양한 크기의 객체 인식 필요, 기존 방식은 많은 메모리와 엄청난 양의 연산을 통해 이루어졌습니다. => 여러 방면에서 상당히 비효율적이므로 이를 개선하고자 나타난 방식인 FPN(Feature Pyramid Net)을 도입하기로 결정하였습니다.
FPN과와 첫 번째 논문의 아이디어인 어텐션 모듈의 결합 코드를 베이스 라인 코드에 이식했을 때 성능이 높아질 것이라는 아이디어와 구조도를 초록으로 작성 후 학회에서 공유 후 연구를 종료하였습니다.
