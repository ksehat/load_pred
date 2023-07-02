import boto3
import numpy as np

# Set the name of your S3 bucket and the prefix for your data files
bucket_name = 'baggage-train-data'
data_prefix = 'data'

# Set your AWS access key ID and secret access key
aws_access_key_id = 'AKIA44R3S2RDU2NFMXWB'
aws_secret_access_key = '22MJss0V6403X1eKhpKuUUBoBdvoOjOh7g2MwLfe'

# Create an S3 client with your credentials
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
response = s3.list_buckets()

# Read the data from the text files and convert it to NumPy arrays
x_train = np.loadtxt('x_train.csv', delimiter=',')
x_test = np.loadtxt('x_test.csv', delimiter=',')
y_train = np.loadtxt('y_train.csv', delimiter=',')
y_test = np.loadtxt('y_test.csv', delimiter=',')

# Save the NumPy arrays to temporary files in binary format
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


# s3.create_bucket(Bucket=bucket_name)
# Upload the temporary files to S3
s3.upload_file('x_train.npy', bucket_name, f'{data_prefix}/x_train.npy')
s3.upload_file('y_train.npy', bucket_name, f'{data_prefix}/y_train.npy')
s3.upload_file('x_test.npy', bucket_name, f'{data_prefix}/x_test.npy')
s3.upload_file('y_test.npy', bucket_name, f'{data_prefix}/y_test.npy')
