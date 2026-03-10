import parser.py


class Music:
    def __init__(self, file_path):
        self.file_path = file_path
        self.score = parser.parse_midi(file_path)

    def get_notes(self):
        return self.parser.get_notes()