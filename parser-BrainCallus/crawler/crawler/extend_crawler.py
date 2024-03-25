from selenium.webdriver.chrome.options import Options
from urllib3.exceptions import MaxRetryError

from crawler.crawler import *


class ExtendedGovCrawler(RegulationGovCrawler):
    month_to_number = {
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
        'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
    }

    project_page_uri_prefix = RegulationGovCrawler.host + '/Regulation/Npa/PublicView?npaID='

    def extract_content(self, testset_name='', start_date=None, end_date=None, period=None, testset_limit=-1):
        super().process_crawl_pages(start_date, end_date, period, testset_limit)
        writer = super().write_standard_project_data(testset_name)
        self.crawl_project_pages()
        if isinstance(writer, ExelWriter):
            writer.write_single(0, 7, 'creation date')
            writer.write_single(0, 8, 'author')
            writer.write(list(map(lambda x: [x.creation_date, x.author], super().get_valid_contents())), start_column=7)
            writer.close()

    def crawl_project_pages(self):
        start = datetime.now()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.run_gathering_extend(super().get_valid_contents()))
        print('\033[52;96m Total time execution to collect extend: \033[0m  ' + str(datetime.now() - start))

    async def run_gathering_extend(self, projects):
        with ThreadPoolExecutor(10) as executors:
            print(f'\033[52;96m extracting author and date for {projects.__len__()} projects:   \033[0m', end='')
            for r in executors.map(self.crawl_project_page, projects):
                print('*')  # log for track progress
            print('')
            executors.shutdown()

    def crawl_project_page(self, project):
        href = self.project_page_uri_prefix + str(project.project_id)
        creation_date = None
        author = ''
        try:
            chrome_option = Options()
            chrome_option.add_argument("--use-fake-ui-for-media-stream")
            chrome_option.add_argument("--disable-user-media-security=true")
            with webdriver.Chrome(chrome_option) as driver:
                driver.get(href)
                driver.implicitly_wait(5)
                timer = 5
                while driver.find_elements(By.CLASS_NAME, 'btns-group').__len__() == 0:
                    self.log_info('Waiting load project page ' + href + ' for ' + str(timer) + 's')
                    timer = timer + 5
                    if timer > 50:
                        self.log_error('Can\'t get data from ' + href + ': timelimit waiting exceeded')
                        return
                    driver.implicitly_wait(5)

                driver.execute_script("$(\".btns-group a\")[0].click()")
                dd = driver.find_elements(By.TAG_NAME, 'dd')
                if dd.__len__() > 2:
                    creation_date = self.extract_date(dd[2].text)
                    author = dd[4].text

        except MaxRetryError:
            self.log_error("Fail to visit " + href)
        except JavascriptException:
            self.log_error("Fail execute script on " + href)
        except WebDriverException:
            self.log_error("Fail to visit " + href)

        project.set_author(author)
        project.set_creation_date(None if creation_date is None else self.get_date_as_string(creation_date))

    def extract_date(self, date_str):
        args = date_str.split(' ')
        if self.month_to_number.keys().__contains__(args[1]):
            return date(int(args[2]), self.month_to_number.get(args[1]), int(args[0]))
        return None
