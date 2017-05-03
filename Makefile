ifdef GPU
	DOCKER_CMD = nvidia-docker
	DOCKER_TAG = latest
else
	DOCKER_CMD = docker
	DOCKER_TAG = nogpu
endif

build:
	cd containers/gpu && docker build -t naoys/shogi-camera:latest .
build-nogpu:
	cd containers/nogpu && docker build -t naoys/shogi-camera:nogpu .
notebook:
	$(DOCKER_CMD) run -v $(PWD):/app -p 8888:8888 -e "JUPYTER_CONFIG_DIR=/app/config" -it naoys/shogi-camera:$(DOCKER_TAG) jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
