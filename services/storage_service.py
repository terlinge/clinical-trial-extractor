"""
PDF Storage Service - Handles storage in filesystem or S3
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError
from flask import current_app

class PDFStorageService:
    def __init__(self):
        self.storage_type = current_app.config.get('PDF_STORAGE_TYPE', 'filesystem')
        
        if self.storage_type == 's3':
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=current_app.config.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=current_app.config.get('AWS_SECRET_ACCESS_KEY'),
                region_name=current_app.config.get('AWS_REGION', 'us-east-1')
            )
            self.bucket_name = current_app.config.get('S3_BUCKET_NAME')
            if not self.bucket_name:
                raise ValueError("S3_BUCKET_NAME must be set when using S3 storage")
        else:
            self.storage_path = Path(current_app.config.get('PDF_STORAGE_PATH', 'pdf_storage'))
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_pdf(self, file_content: bytes, filename: str = None) -> Tuple[str, str]:
        """
        Store PDF and return (storage_path, file_hash)
        """
        # Calculate hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        if self.storage_type == 's3':
            key = f'pdfs/{file_hash}.pdf'
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=file_content,
                    ContentType='application/pdf',
                    Metadata={
                        'original_filename': filename or 'unknown.pdf',
                        'hash': file_hash
                    }
                )
                storage_path = f's3://{self.bucket_name}/{key}'
            except ClientError as e:
                raise Exception(f"Failed to upload PDF to S3: {str(e)}")
        else:
            file_path = self.storage_path / f'{file_hash}.pdf'
            try:
                file_path.write_bytes(file_content)
                storage_path = str(file_path.absolute())
            except IOError as e:
                raise Exception(f"Failed to save PDF to filesystem: {str(e)}")
        
        return storage_path, file_hash
    
    def retrieve_pdf(self, storage_path: str) -> bytes:
        """
        Retrieve PDF content from storage
        """
        if self.storage_type == 's3' and storage_path.startswith('s3://'):
            # Parse S3 path
            path_parts = storage_path.replace('s3://', '').split('/', 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else ''
            
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    raise FileNotFoundError(f"PDF not found in S3: {storage_path}")
                raise Exception(f"Failed to retrieve PDF from S3: {str(e)}")
        else:
            try:
                return Path(storage_path).read_bytes()
            except FileNotFoundError:
                raise FileNotFoundError(f"PDF not found: {storage_path}")
            except IOError as e:
                raise Exception(f"Failed to read PDF from filesystem: {str(e)}")
    
    def delete_pdf(self, storage_path: str) -> bool:
        """
        Delete PDF from storage
        """
        try:
            if self.storage_type == 's3' and storage_path.startswith('s3://'):
                path_parts = storage_path.replace('s3://', '').split('/', 1)
                bucket = path_parts[0]
                key = path_parts[1] if len(path_parts) > 1 else ''
                self.s3_client.delete_object(Bucket=bucket, Key=key)
            else:
                Path(storage_path).unlink()
            return True
        except Exception as e:
            print(f"Failed to delete PDF: {str(e)}")
            return False
    
    def generate_signed_url(self, storage_path: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for S3 objects (returns None for filesystem storage)
        """
        if self.storage_type != 's3' or not storage_path.startswith('s3://'):
            return None
        
        path_parts = storage_path.replace('s3://', '').split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Failed to generate signed URL: {str(e)}")
            return None