import json
import requests
import time
import zipfile
import io
import base64
import csv
import assemblyai as aai

def merge_diarization_and_transcription(dia_segments, words):
    print(dia_segments)
    first_speaker = dia_segments[0]["speaker"]
    speaker_map = {first_speaker: "parent"}
    for segment in dia_segments:
        s = segment["speaker"]
        if s not in speaker_map:
            speaker_map[s] = "child"
    result = []
    for word in words:
        word_start = word.start/1000
        word_end = word.end/1000
        word_text = word.text

        speaker = "Unknown"
        for segment in dia_segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_speaker = segment.get("speaker", "Unknown")

            if seg_start - 0.1 <= word_start <= seg_end + 0.1:
                speaker = speaker_map.get(seg_speaker, "Unknown")
                break

        result.append({
            "speaker": speaker,
            "start": word_start,
            "end": word_end,
            "text": word_text
        })
    return result

def lambda_handler(event, context):
    FILE_URLS = [
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+01.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+02.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+03.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+04.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+05.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+06.m4a",
        "https://ratestaudio.s3.us-east-1.amazonaws.com/Case+07.m4a"
    ]

    dia_results = {}
    trans_results = {}
    aai.settings.api_key = "73b8c935469a402b9a7f6c02800baf88"
    for file_url in FILE_URLS:
        # pyannote diarization
        payload = {
            "url": file_url,
            "numSpeakers": 2,
            "confidence": False
        }
        headers = {
            "Authorization": "Bearer sk_dc94afdca80b4ddcb9f5feb593d5e29f",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", "https://api.pyannote.ai/v1/diarize", json=payload, headers=headers)

        job_data = response.json()
        job_id = job_data.get("jobId")


        # poll until ready
        result_url = f"https://api.pyannote.ai/v1/jobs/{job_id}"
        while True:

            result_response = requests.get(result_url, headers=headers)
            result_data = result_response.json()
  

            if result_data.get("status") == "succeeded":
 
                dia_results[file_url] = result_data["output"]["diarization"]
                break
            elif result_data.get("status") == "failed":
                dia_results[file_url] = []
                break
            time.sleep(1)

        # AssemblyAI transcription
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_url)

        if transcript.status == aai.TranscriptStatus.error:
            trans_results[file_url] = []
        else:
            trans_results[file_url] = transcript.words

    # write zip with CSVs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_url in FILE_URLS:
            file_name = file_url.split("/")[-1].replace(".wav", "").replace(".m4a", "").replace("+", "_")
            merged_rows = merge_diarization_and_transcription(dia_results[file_url], trans_results[file_url])

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["speaker", "start", "end", "text"])
            writer.writeheader()
            for row in merged_rows:
                writer.writerow(row)

            zip_file.writestr(file_name + ".csv", csv_buffer.getvalue())

    zip_buffer.seek(0)
    zip_data = zip_buffer.getvalue()
    encoded_zip = base64.b64encode(zip_data).decode("utf-8")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/zip",
            "Content-Disposition": "attachment; filename=results.zip"
        },
        "body": encoded_zip,
        "isBase64Encoded": True
    }
