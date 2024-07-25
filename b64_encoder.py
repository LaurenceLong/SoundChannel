import base64


def file_to_base64(file_path):
    with open(icon_path, "rb") as temp_file:
        encoded_string = base64.b64encode(temp_file.read())
    return encoded_string.decode('utf-8')


if __name__ == "__main__":
    # 使用方法
    icon_path = "./icon.png"  # 或 .png, .jpg 等
    base64_string = file_to_base64(icon_path)
    with open("b64_encoded_files.py", "w+") as fd:
        fd.write(f'icon_base64 = "{base64_string}"')
