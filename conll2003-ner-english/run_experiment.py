import click

from utils_experiment import ExperimentRunner


@click.command()
@click.option("--config", type=str, help="Define path to configuration file")
@click.option("--number", type=int, help="Define experiment number")
def run_experiment(config, number):
    runner = ExperimentRunner(number=number, configuration_file=config)
    runner.start()


if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
