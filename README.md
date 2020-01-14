# machinelearningpipeline
Data Pipeline to pull data, clean it and run an ML algorithm on it.
Install Airflow locally using pip install apache-airflow.

Then create a dag folder to create airflow directory and copy all the dags.
then run the following commands
airflow initdb
airflow webserver -d
airflow scheduler

Once you have this.

Run docker-compose up --build

This should create 3 images ml_base, arangodb and mongodb

This will then start all the process in the DAG:

1. Pull data from CSV and dump in the Mongo DB Collection
2. Then Advisor will pull this data.
3. Then advisor run the ML model on the data and generates the recommendations.
4. This recommendations gets saved in the arangodb.
5. You can change the connection details in runtime in arango collection called templates.


