IMAGE_NAME=depoco
TAG=latest
DATASETS=/home/rashmi/Documents/data_for_depoco/submaps/40m_ILEN
OUTPUT_DIR=/home/rashmi/Documents/depoco_output

build:
	@echo Building docker container $(IMAGE_NAME)
	docker build -t $(IMAGE_NAME):$(TAG) .

test:
	@echo NVIDIA and CUDA setup
	@docker run --rm $(IMAGE_NAME):$(TAG) nvidia-smi
	@echo PytTorch CUDA setup installed?
	@docker run --rm $(IMAGE_NAME):$(TAG) python3 -c "import torch; print(torch.cuda.is_available())"

run:
	docker run --rm --gpus all -p 8888:8888 -it -e DISPLAY=$(DISPLAY) -e  "XDG_RUNTIME_DIR" -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(XDG_RUNTIME_DIR):$(XDG_RUNTIME_DIR)" -v $(DATASETS):/data -v $(OUTPUT_DIR):/output $(IMAGE_NAME) 
  	

clean:
	@echo Removing docker image...
	-docker image rm --force $(IMAGE_NAME):$(TAG)
