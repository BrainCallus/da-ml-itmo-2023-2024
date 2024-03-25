from crawler.extend_crawler import *


def run_crawler():
    cr = RegulationGovCrawler()
    cr.extract_content(testset_name='base_30.xlsx', start_date=date(2024, 2, 1), end_date=date(2024, 3, 1))
    print("Total size of data collected " + str(cr.get_result()))


def run_extend_crawler():
    print(
        '\033[98;93m [WARN]\033[0m site https://regulation.gov.ru is really slow, ' +
        'gathrering information from single page may take up to 1 minute. Please wait')
    cr = ExtendedGovCrawler()
    cr.extract_content(testset_name='extend_3days.xlsx', period=timedelta(days=3))


if __name__ == '__main__':
    run_crawler()
    # short example of work ExtendedGovCrawler;
    # it processed 60 pages within 12 minutes therefore I decided not to take a period longer than 3 days for tests
    # run_extend_crawler()
