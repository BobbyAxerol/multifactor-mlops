FROM python:3.12-slim  
  
WORKDIR /app  
  
RUN pip install --no-cache-dir \  
    pandas \  
    numpy \  
    xgboost  
  
COPY src/multifactor_mlops /app/src/multifactor_mlops  
COPY parameters.json /app/parameters.json  
COPY artifacts/model_bundle /app/artifacts/model_bundle  
  
EXPOSE 8000  
  
CMD ["python", "-c", "import sys; sys.path.insert(0, '/app'); from src.multifactor_mlops.pipelines.fit_final import ProductionModelBundle; b = ProductionModelBundle.from_dir('/app/artifacts/model_bundle'); print('bundle OK, features=', len(b.feature_names))"]  
