# -*- coding: utf-8 -*-
import asyncio
import functools
import threading
from datetime import *

from selenium import webdriver
from selenium.common import StaleElementReferenceException, JavascriptException, WebDriverException
from selenium.webdriver.common.by import By

from LegislativeDraft import *
from ExelWriter import *
from concurrent.futures import ThreadPoolExecutor


def synchronized(wrapped):
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            result = wrapped(*args, **kwargs)
            return result

    return _wrap


def negate(other_to_bool_lambda):
    return lambda x: not other_to_bool_lambda(x)


class RegulationGovCrawler:
    host = "https://regulation.gov.ru"
    total_time_execution = 0
    available_departments = ['Минобороны России', 'Росавиация', 'МВД России', 'ФСИН России', 'ФСБ России', 'Росгвардия',
                             'Следственный комитет Российской Федерации']

    def __init__(self):
        self.driver = None
        self.contents = []
        self.achieved = False
        self.start_date = None
        self.end_date = None
        self.testset_limit = -1
        self.corrupted_count = 0

    def extract_content(self, testset_name='', start_date=None, end_date=None, period=None, testset_limit=-1):
        self.process_crawl_pages(start_date, end_date, period, testset_limit)
        self.write_standard_project_data(testset_name).close()

    def process_crawl_pages(self, start_date=None, end_date=None, period=None, testset_limit=-1):
        self.drop_dates()
        if testset_limit > 0:
            self.testset_limit = testset_limit
        if self.verify_and_set_period_date(start_date, end_date, period):
            self.crawl_pages(self.start_date, self.end_date)

    def drop_dates(self):
        self.start_date = None
        self.end_date = None
        self.corrupted_count = 0

    def verify_and_set_period_date(self, start_date=None, end_date=None, period=None):
        if isinstance(start_date, date) and isinstance(start_date, date):
            if end_date > start_date:
                self.start_date = start_date
                self.end_date = end_date
            else:
                self.log_error('Can\'t process crawling. Delta between dates should be positive days amount')
                return False
        elif isinstance(period, timedelta):
            self.start_date = datetime.today() - period
            self.end_date = date.today()
        else:
            self.log_error('Can\'t process crawling. Expected 2 dates or period')
            return False
        return True

    def write_standard_project_data(self, testset_name=''):
        if testset_name == '':
            testset_name = self.get_default_testset_name()
        writer = ExelWriter(testset_name)
        writer.set_title_row(
            ['projectId', 'department', 'document type', 'title', 'progress', 'views', 'project card size Kb'])
        writer.write(list(map(lambda x: x.to_exel_row_str(), self.get_valid_contents())))
        return writer

    def crawl_pages(self, start_date, end_date):
        start = datetime.now()
        suffix = '#StartDate=' + self.get_date_as_string(
            start_date) + '&EndDate=' + self.get_date_as_string(end_date)

        self.driver = webdriver.Chrome()
        try:
            self.driver.get(self.host + '/projects' + suffix)
            self.driver.implicitly_wait(10)
            content_amount_info = ''
            while content_amount_info == '':
                content_amount_info = self.driver.find_element(By.CLASS_NAME, 'k-pager-info.k-label').text

            max_content = int(content_amount_info.split(' ')[4])
            total_pages = int(max_content / 20) + (1 if max_content % 20 != 0 else 0)
            self.crawl_project_list(1, total_pages)

        except JavascriptException as js:
            self.log_error(
                'Something gone wrong during execution, possibly not all data collected: JavascriptException occurred:\n' + str(
                    js)
            )
        except WebDriverException as w:
            self.log_error(
                'Something gone wrong during execution, possibly not all data collected: WebDriverException occurred:\n' + str(
                    w)
            )
        finally:
            print('\033[52;96m Total time execution to collect base: \033[0m  ' + str(datetime.now() - start))
            self.driver.close()

    def get_result(self):

        return self.contents.__len__()

    @staticmethod
    def get_date_as_string(date_date):
        return '-' if date_date is None else '.'.join([str(date_date.day), str(date_date.month), str(date_date.year)])

    def get_valid_contents(self):
        return list(filter(negate(LegislativeDraft.is_corrupted), self.contents))

    def get_progress(self, project_status):
        if project_status is None:
            return 0
        elif self.safe_get_html(project_status, By.TAG_NAME, 'span').text == 'Обсуждение завершено':
            return 1.0
        else:
            background = project_status.get_attribute('style')
            if background is None:
                return 0
            else:
                try:
                    p = background.split("linear-gradient(")[1].split(", ")[3].split(' ')[1]
                    return float(p[:-1]) / 100.0
                except IndexError as idx:
                    self.log_error('Failed to get completeness:\n' + str(idx))
                except ValueError as v:
                    self.log_error('Failed to get float getting completeness:\n' + str(v))
            return 0.0

    def crawl_project_list(self, current_list, total_pages):
        projects = []
        tries = 1
        while projects.__len__() == 0 and tries < 15:
            self.driver.implicitly_wait(3)
            tries = tries + 1
            self.log_info('Trying get projects from list ' + str(current_list) + '.. at ' + str(tries) + ' time')
            projects = self.driver.find_elements(By.CLASS_NAME, 'project')

        if projects.__len__() > 0:
            self.driver.implicitly_wait(5)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.run_gathering(projects))

        else:
            self.log_error('Failed to get projects from ' + str(current_list) + ' list. Go to next')

        if current_list < total_pages and not self.achieved:
            self.driver.execute_script(
                '$(".k-pager-numbers.k-reset a[data-page={next_page:d}]").click()'.format(next_page=current_list + 1))
            self.crawl_project_list(current_list + 1, total_pages)

    async def run_gathering(self, projects):
        with ThreadPoolExecutor(projects.__len__()) as executors:
            print('\033[52;96m getting results:   \033[0m', end='')
            for r in executors.map(self.handle_project, projects):
                print('*', end='')
            print('')
            executors.shutdown()

    def handle_project(self, project_bar):
        def try_get_text(element, class_name):
            val = ''
            i = 0
            while val == '' and i < 15:
                val = self.safe_get_html(element, By.CLASS_NAME, class_name).text
                i = i + 1
            return val

        title_block = self.safe_get_html(project_bar, By.CLASS_NAME, 'project-title__block')
        department = try_get_text(title_block, 'project-title__block_title')
        document_type = try_get_text(title_block, 'project-title__block_subtitle')

        project_title = self.safe_get_html(self.safe_get_html(project_bar, By.CLASS_NAME, 'project-body'), By.TAG_NAME,
                                           'a')
        project_id = -1
        if project_title is not None:
            project_id = self.get_int_or_else(project_title.get_attribute('data-id'), -1)
            project_title = project_title.get_attribute('title')

        progress = self.get_progress(self.safe_get_html(project_bar, By.CLASS_NAME, 'project-body__status'))
        footer_views = self.safe_get_html(self.safe_get_html(project_bar, By.CLASS_NAME, 'project-footer__group-2')
                                          , By.TAG_NAME, 'span')
        views = 0 if (footer_views is None) else self.get_int_or_else(footer_views.text, 0)
        card_size = float(len(project_bar.text.encode('utf-8')) / 1024.0)

        if 0 < self.testset_limit <= self.get_current_valid_size():
            self.mark_achieved()
            return 'Got nothing. Project with id={id:} skipped maximum testset size exceed'.format(id=project_id)
        else:
            project_data = LegislativeDraft(project_id, department, document_type, project_title, progress, views,
                                            card_size)
            self.add_project_data(project_data)
            return project_data.to_string()

    def safe_get_html(self, element, by, query):
        to_return = None
        try:
            founded = [] if (element is None) else element.find_elements(by, query)
            if founded.__len__() > 0:
                to_return = founded[0]
        except StaleElementReferenceException:
            self.log_error('\033[31;1m Failed to get element. Go to next \033[0m')
        return to_return

    def get_int_or_else(self, string, on_fault):
        try:
            res = int(string)
            return res
        except ValueError as err:
            self.log_error('Failed to get int from ' + string + ':\n' + str(err))
            return on_fault

    @staticmethod
    def get_default_testset_name():
        today = datetime.today()
        return 'test__' + '.'.join([str(datetime.today().day), str(today.month), str(today.year), str(today.hour) + 'h',
                                    str(datetime.now().minute) + 'm']) + '.xlsx'

    @synchronized
    def add_project_data(self, data):
        if data.project_id == -1:
            self.corrupted_count = self.corrupted_count + 1
        self.contents.append(data)

    @synchronized
    def get_current_valid_size(self):
        return self.contents.__len__() - self.get_corrupted_count()

    @synchronized
    def get_corrupted_count(self):
        return self.corrupted_count

    @synchronized
    def mark_achieved(self):
        self.achieved = True

    @staticmethod
    def print_log(message, tag):
        print(tag + message)

    def log_error(self, message):
        self.print_log(message, '\033[31;1m [ERROR] \033[0m')

    def log_info(self, message):
        self.print_log(message, '\033[98;93m [INFO] \033[0m')
