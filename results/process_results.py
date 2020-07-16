# std
import os
import sys
import re
import csv
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(__file__))
    logger.debug(f'root: {root}')

    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    logger.debug(f'path: {path}')

    return path


def parse_results(path):
    with open(path, 'r') as resultsfile:
        pattern = re.compile(r'(Epoch: [0-9]+), (([a-zA-Z0-9\s@]+)_[a-z]+: [0-9]+\.?[0-9]*)')
        results_cost = defaultdict(list)
        results_hits_at_10 = defaultdict(list)
        results_hits_at_3 = defaultdict(list)
        results_hits_at_1 = defaultdict(list)
        results_mean_rank = defaultdict(list)
        results_mean_reciprocal_rank = defaultdict(list)

        for line in resultsfile:
            line = line.strip()
            logger.debug(f'line: {line}')
            record = pattern.findall(line)
            logger.debug(f'record: {record}')

            if record:
                record, = record
                metric = record[2]
                logger.debug(f'metric: {metric}')
                value = float(record[1].split(':')[1].strip())
                logger.debug(f'value: {value}')

                epoch = int(record[0].split(':')[1])
                if epoch % 10 == 0:
                    if metric == 'Mean evaluation cost':
                        results_cost[record[0]].append(value)
                        logger.debug(f'results_cost: {results_cost}')
                    elif metric == 'Hits @10':
                        results_hits_at_10[record[0]].append(value)
                        logger.debug(f'results_hits_at_10: {results_hits_at_10}')
                    elif metric == 'Hits @3':
                        results_hits_at_3[record[0]].append(value)
                        logger.debug(f'results_hits_at_3: {results_hits_at_3}')
                    elif metric == 'Hits @1':
                        results_hits_at_1[record[0]].append(value)
                        logger.debug(f'results_hits_at_1: {results_hits_at_1}')
                    elif metric == 'Mean rank':
                        results_mean_rank[record[0]].append(value)
                        logger.debug(f'results_mean_rank: {results_mean_rank}')
                    elif metric == 'Mean reciprocal rank':
                        results_mean_reciprocal_rank[record[0]].append(value)
                        logger.debug(f'results_mean_reciprocal_rank: {results_mean_reciprocal_rank}')

    logger.info(f'results_cost: {results_cost}')
    logger.info(f'results_hits_at_10: {results_hits_at_10}')
    logger.info(f'results_hits_at_3: {results_hits_at_3}')
    logger.info(f'results_hits_at_1: {results_hits_at_1}')
    logger.info(f'results_mean_rank: {results_mean_rank}')
    logger.info(f'results_mean_reciprocal_rank: {results_mean_reciprocal_rank}')

    return results_cost, \
           results_hits_at_10, \
           results_hits_at_3, \
           results_hits_at_1, \
           results_mean_rank, \
           results_mean_reciprocal_rank


def write_results(results, data_set):
    for metric in results:
        metric_baseline, metric_hypothesis = results[metric]

        with open(f'hntn_train_validate_and_test_{data_set}_200d_{metric}.csv', mode='w') as resultsfile:
            csv_writer = csv.writer(resultsfile)
            csv_writer.writerow([f'{metric}_training_hypothesis',
                                 f'{metric}_validation_hypothesis',
                                 f'{metric}_test_hypothesis',
                                 f'{metric}_training_baseline',
                                 f'{metric}_validation_baseline',
                                 f'{metric}_test_baseline'])

            for epoch in metric_hypothesis:
                result = metric_hypothesis[epoch]
                result.extend(metric_baseline[epoch])
                logger.debug(f'result: {result}')
                csv_writer.writerow(result)


def main(data_set):
    path_baseline = get_path(f'hntn_train_validate_and_test_{data_set}_200d_baseline.log', 'results')
    path_hypothesis = get_path(f'hntn_train_validate_and_test_{data_set}_200d_hypothesis.log', 'results')

    logger.info('Parsing baseline results...')
    results_cost_baseline, \
    results_hits_at_10_baseline, \
    results_hits_at_3_baseline, \
    results_hits_at_1_baseline, \
    results_mean_rank_baseline, \
    results_mean_reciprocal_rank_baseline = parse_results(path_baseline)
    logger.info('Parsing baseline results complete!')

    logger.info('Parsing hypothesis results...')
    results_cost_hypothesis, \
    results_hits_at_10_hypothesis, \
    results_hits_at_3_hypothesis, \
    results_hits_at_1_hypothesis, \
    results_mean_rank_hypothesis, \
    results_mean_reciprocal_rank_hypothesis = parse_results(path_hypothesis)
    logger.info('Parsing hypothesis results complete!')

    logger.info('Writing results...')
    results = {'cost': (results_cost_baseline, results_cost_hypothesis),
               'hits_at_10': (results_hits_at_10_baseline, results_hits_at_10_hypothesis),
               'hits_at_3': (results_hits_at_3_baseline, results_hits_at_3_hypothesis),
               'hits_at_1': (results_hits_at_1_baseline, results_hits_at_1_hypothesis),
               'mean_rank': (results_mean_rank_baseline, results_mean_rank_hypothesis),
               'mean_reciprocal_rank': (results_mean_reciprocal_rank_baseline, results_mean_reciprocal_rank_hypothesis)}
    logger.debug(f'results: {results}')
    write_results(results, data_set)
    logger.info('Writing results complete!')


if __name__ == '__main__':
    logger.info('START!')
    main(data_set='fb15k_237')
    logger.info('DONE!')
