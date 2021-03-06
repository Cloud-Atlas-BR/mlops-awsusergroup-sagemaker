import json
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

session = sagemaker.Session()
region_name = session.boto_region_name

container_image = "811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"

bucket = "sagemaker-us-east-1-885248014373"

role = "arn:aws:iam::885248014373:role/service-role/AmazonSageMaker-ExecutionRole-20210305T230941"

uploaded_data = "s3://sagemaker-us-east-1-885248014373/training/data.csv"

input_data = TrainingInput(s3_data=uploaded_data,
                                            content_type="text/csv")

xgboost = Estimator(
    image_uri=container_image,
    role=role,
    instance_type="ml.m5.large", 
    instance_count=1,
    output_path=f"s3://{bucket}/output",
    sagemaker_session=session)   

xgboost.set_hyperparameters(num_round=5, max_depth=5)

xgboost.fit({"train": input_data}) 

model_data = xgboost.model_data
print(model_data)

with open("../deploy/parameters.json", "r") as f:
    parameters = json.load(f)

parameters["Parameters"]["ModelDataUrl"] = model_data

with open("../deploy/parameters.json", "w") as f:
    json.dump(parameters, f)