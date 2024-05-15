down:
	docker-compose -f docker/docker-compose.yml down --remove-orphans 
up:
	docker-compose -f docker/docker-compose.yml up --build --remove-orphans -d
up-f:
	docker-compose -f docker/docker-compose.yml up --build --remove-orphans