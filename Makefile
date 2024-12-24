start-ollama:
	@if [ "$$(docker container ls -f Name=ollama -q)" ]; then \
		echo "ollama is already running"; \
	else \
		docker run -d --gpus=all --network host --rm -v /mnt/Data/ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama; \
	fi

start-letta: start-ollama
	@mkdir -p letta-workspace && \
	cd letta-workspace && \
	OLLAMA_BASE_URL=http://localhost:11434 letta server > server.log 2>&1 & \
	if [ $$? -eq 0 ]; then \
		echo "start letta success"; \
	else \
		echo "start letta failed"; \
		docker container stop ollama; \
	fi

stop-letta:
	@kill $$(ps aux | grep "letta server" | grep -v grep | awk '{print $$2}') | xargs -I % kill %

clear-letta:
	@rm -rf letta-workspace/* ~/.letta/*

start-open-webui: start-ollama
	docker run -d \
	--network host \
	-e PORT=8888 \
	--gpus all \
	-v $$HOME/open-webui:/app/backend/data \
	--name open-webui \
	--rm ghcr.io/open-webui/open-webui:cuda
