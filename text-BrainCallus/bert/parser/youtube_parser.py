import csv
from googleapiclient.discovery import build
from bert.bert_model.logger import Logger


class YoutubeParser:
    def __init__(self, api_key):
        self.api_key = api_key
        self.comments = []
        self.next_page_token = None
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.logger = Logger(YoutubeParser.__name__)

    def parse_comments(self, video_id):
        self.logger.info(f'Start parse comments for {video_id}')
        while True:
            response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=self.next_page_token
            ).execute()

            for item in response['items']:
                comment_info = item['snippet']['topLevelComment']['snippet']
                author = comment_info['authorDisplayName']
                c_time = comment_info['publishedAt']
                date = c_time.split('T')[0]
                text = comment_info['textDisplay']
                self.comments.append(self.CommentEntry(author, date, c_time, text))

            self.next_page_token = response.get('nextPageToken')
            if not self.next_page_token:
                self.logger.info(f'Finished parse comments for video: {video_id}. Parsed {len(self.comments)} comments')
                break

    def write_csv(self, file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['author', 'date', 'time', 'comment'])
            writer.writeheader()
            for comment in self.comments:
                writer.writerow(comment.to_row())
            self.logger.info(f'Comments were successfully written to {file_path}')

    class CommentEntry:
        def __init__(self, author, date, tme, text):
            self.author = author
            self.date = date
            self.tme = tme
            self.text = text

        def to_row(self):
            return {'author': self.author, 'date': self.date, 'time': self.tme, 'comment': self.text}
