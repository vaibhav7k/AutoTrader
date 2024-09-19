# Use AWS Lambda Python 3.8 image
FROM public.ecr.aws/lambda/python:3.8

# Set working directory
WORKDIR /var/task

# Copy everything from the current directory to /var/task
COPY . .

# Upgrade pip (optional but recommended)
RUN pip install --upgrade pip

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Specify the Lambda handler
CMD ["lambda_function.lambda_handler"]
