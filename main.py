from imageai.Detection.Custom import DetectionModelTrainer
import yaml

with open("riichiDataset/data.yaml", "r") as yaml_file:
    data = yaml.safe_load(yaml_file)




trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="riichiDataset")
trainer.setTrainConfig(object_names_array=data["names"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
trainer.trainModel()


