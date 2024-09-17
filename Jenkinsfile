pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/seu-usuario/seu-repositorio.git'
            }
        }

        stage('Install dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'python -m unittest discover -s tests'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python src/model.py'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'python src/evaluate.py'
            }
        }

        stage('Deploy Model') {
            steps {
                sh 'docker build -t churn-model-deploy .'
                sh 'docker run -d -p 5000:5000 churn-model-deploy'
            }
        }
    }

    post {
        always {
            junit '**/test-reports/*.xml'
        }
    }
}
