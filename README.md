# Running this Application with Docker
1. Clone this repository to your local machine:
   
   ```bash
   git clone https://github.com/KonBes/texnlogismikou.git
   cd texnlogismikou
2. Build the Docker Image:
   
   ```bash
   docker build -t streamlit-app .
3. Run the Docker container:
   
   ```bash
   docker run -p 8501:8501 streamlit-app
  
4. Open your browser and go to http://localhost:8501 to access the application.
