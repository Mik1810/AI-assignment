docker stop ai_project_module
docker rm ai_project_module
docker rmi ai_project_image
docker build -t ai_project_image .
docker run -d --name ai_project_module -p 8080:8080 ai_project_image
docker ps
