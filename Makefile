init:
	mkdir data

	curl -L -o ./data/RMDaily.zip \
	 https://www.kaggle.com/api/v1/datasets/download/concyclics/renmindaily

	unzip ./data/RMDaily.zip -d ./data && mv ./data/RenMin_Daily.csv ./data/RMDaily.csv && rm ./data/RMDaily.zip

	docker run -p 6333:6333 -p 6334:6334 \
    	-v $(pwd)/qdrant:/qdrant/storage:z \
    	qdrant/qdrant

build:
	cargo b -r