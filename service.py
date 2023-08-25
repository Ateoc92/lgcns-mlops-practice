import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    bhk: int
    size: int
    floor: str
    area_type: str
    city: str
    furnishing_status: str
    tenant_preferred: str
    bathroom: int
    point_of_contact: str


# 학습 코드에서 저장한 베스트 모델을 가져올 것 (house_rent:latest)
# https://github.com/bentoml/BentoML/blob/main/examples/sklearn/linear_regression/service.py 참고

bento_model = bentoml.sklearn.get("house_rent:latest")
model_runner = bento_model.to_runner()

# "rent_house_regressor"라는 이름으로 서비스를 띄우기
svc = bentoml.Service("rent_gradregressor", runners=[model_runner])


@svc.api(
    # TODO: Features 클래스를 JSON으로 받아오고 Numpy NDArray를 반환하도록 데코레이터 작성
    input=JSON(pydantic_model=Features),
    output=NumpyNdarray(),
)
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(input_df)  # 로그변환했음...
    return np.expm1(log_pred)
