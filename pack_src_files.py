import os
import zipfile
from datetime import datetime

# 您已有的文件列表
src_files = [
    '.gitignore',
    'amodem',
    'amodem_cmd.txt',
    # 'b64_encoded_files.py',
    'b64_encoder.py',
    'build_cmd.txt',
    # 'icon.ico',
    # 'icon.png',
    'pack_src_files.py',
    'requirements.txt',
    'sound_channel.py',
    'sound_channel_ui.py',
    'test.txt',
]


def zip_files(file_list, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_list:
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            elif os.path.isdir(file):
                for root, dirs, files in os.walk(file):
                    # 排除 __pycache__ 文件夹
                    dirs[:] = [d for d in dirs if d != '__pycache__']
                    for f in files:
                        file_path = os.path.join(root, f)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(file))
                        zipf.write(file_path, arcname)


# 生成一个带有时间戳的 zip 文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
my_zip_filename = f"source_files_{timestamp}.zip"
# 执行打包
zip_files(src_files, my_zip_filename)
print(f"Files have been zipped into {my_zip_filename}")
