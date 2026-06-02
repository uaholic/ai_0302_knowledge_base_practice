from app.infra.config.providers import infra_config
from app.shared.clients import get_minio_client


class MinioGateway:

    @property
    def bucket_name(self):
        return infra_config.minio.bucket_name

    @property
    def image_dir(self):
        return infra_config.minio.minio_img_dir

    def client(self):
        return get_minio_client()

    def build_img_url(self, stem:str,object_name:str):
        """
        生成图片 URL
        """
        protocol = "https" if infra_config.minio.minio_secure else "http"

        return f"{protocol}://{infra_config.minio.endpoint}/{infra_config.minio.bucket_name}/{self.image_dir}/{stem}/{object_name}"


minio_gateway = MinioGateway()

if __name__ == '__main__':
    url = MinioGateway().build_img_url("test", "test.png")
    print(url)


