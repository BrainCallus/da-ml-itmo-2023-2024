class LegislativeDraft:
    def __init__(self, project_id, department, document_type, title, stage, views, card_size):
        self.project_id = -1 if project_id is None else int(project_id)
        self.department = '' if department is None else department
        self.dock_type = '' if document_type is None else document_type
        self.title = '' if title is None else title

        self.progress = 0 if stage is None else stage
        self.views = 0 if views is None else int(views)
        self.card_size = card_size
        self.author = ''
        self.creation_date = ''

    def set_author(self, author):
        self.author = '' if author is None else author

    def set_creation_date(self, creation_date):
        self.creation_date = '-' if creation_date is None else creation_date

    def is_corrupted(self):
        return self.project_id == -1

    def to_string(self):
        return 'project_id = {project_id:d}\ndepartment = {department:s}\ndocument_type = {dock_type:s}\n'.format(
            project_id=self.project_id, department=self.department,
            dock_type=self.dock_type) + 'title = {title:s}\nprogress = {stage:.2f}%'.format(
            title=self.title,
            stage=self.progress * 100) + '\nviews = {views:d}\nproject card size(Kb) = {card_size:.2f}\n'.format(
            views=self.views,
            card_size=self.card_size) + 'author = {author:s}\ncreation_date = {date:s}'.format(
            author=self.author, date=self.creation_date)

    def to_exel_row_str(self):
        return [self.project_id, self.department, self.dock_type, self.title, self.progress, self.views, self.card_size]
