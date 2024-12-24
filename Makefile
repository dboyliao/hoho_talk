OLLAMA_HOST?=127.0.0.1:11434

start-ollama:
	@if [ "$$(docker container ls -f Name=ollama -q)" ]; then \
		echo "ollama is already running"; \
	else \
		docker run -d --gpus=all --network host \
		--rm -v /mnt/Data/ollama:/root/.ollama \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		--name ollama ollama/ollama; \
	fi

stop-ollama:
	@if [ "$$(docker container ls -f Name=ollama -q)" ]; then \
		docker container stop ollama; \
	else \
		echo "ollama is not running"; \
	fi

start-open-webui: start-ollama
	docker run -d \
	--network host \
	-e PORT=8888 \
	--gpus all \
	-v $$HOME/open-webui:/app/backend/data \
	--name open-webui \
	--rm ghcr.io/open-webui/open-webui:cuda
