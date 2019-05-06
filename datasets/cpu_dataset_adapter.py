"""An adapter for the CPU utilization data collected from Emulab PC3000 machines with Collectl."""


import os

import click


@click.command()
@click.argument("inputdir_path", metavar="<inputdir_path>")
@click.argument("outputdir_path", metavar="<outputdir_path>")
@click.argument("window_size", metavar="<window_size>", type=int)
@click.argument("window_step", metavar="<window_step>", type=int)
def main(inputdir_path, outputdir_path, window_size, window_step):
    """Build sequences of average utilization percentage from files in the specified directory."""
    with open(os.path.join(outputdir_path, "normal"), 'w') as normal_file:
        with open(os.path.join(outputdir_path, "abnormal"), 'w') as abnormal_file:
            for inputfile_name in os.listdir(inputdir_path):
                cpu_totls = []
                with open(os.path.join(inputdir_path, inputfile_name)) as input_file:
                    for input_line in input_file:
                        cpu0_totl = int(input_line.strip().split()[10])
                        cpu1_totl = int(input_line.strip().split()[22])
                        cpu_totls.append(int((cpu0_totl + cpu1_totl) / 2))
                for window_start_index in range(0, len(cpu_totls), window_step):
                    cpu_window = [
                        cpu_totls[i]
                        for i in range(
                            window_start_index,
                            min(window_start_index + window_size, len(cpu_totls))
                        )
                    ]
                    if sum(cpu_window) / len(cpu_window) < 50 and cpu_window.count(100) > 0 and \
                            cpu_window.count(100) < 0.75 * len(cpu_window):
                        abnormal_file.write(' '.join([str(cpu) for cpu in cpu_window]) + '\n')
                    else:
                        normal_file.write(' '.join([str(cpu) for cpu in cpu_window]) + '\n')


if __name__ == "__main__":
    main()
