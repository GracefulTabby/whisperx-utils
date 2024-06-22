import glob
import os
from whisperx_utils import transcribe_audio_file
from dotenv import load_dotenv


def main():
    """
    wavファイルを文字起こしして、結果を新しいディレクトリに保存するプログラム。

    .envファイルからHF_TOKENを読み込み、各wavファイルに対して文字起こしを行います。
    """

    # .env ファイルのパスを指定して読み込む
    dotenv_path = ".env"
    load_dotenv(dotenv_path)

    # Hugging Face Secret
    hf_token = os.getenv("HF_TOKEN")

    # ディレクトリ内のwavファイルリストを取得する
    audio_files = glob.glob("./**/*.wav")

    for path in audio_files:
        # ファイルパスの末尾に_transcribeを付与したdirectoryを作成
        output_path = f"{path}_transcribe"

        # 存在チェック
        if os.path.exists(output_path):
            print(f"{path}のファイルをスキップします。")
            continue

        print(path)

        # 文字起こしを実行
        transcribe_audio_file(
            path,
            model="large-v3",
            model_dir="./models",
            align_model="NTQAI/wav2vec2-large-japanese",
            language="ja",
            batch_size=8,
            output_dir=output_path,
            hf_token=hf_token,
            chunk_size=10,
            diarize=True,
        )

    return


if __name__ == "__main__":
    main()
