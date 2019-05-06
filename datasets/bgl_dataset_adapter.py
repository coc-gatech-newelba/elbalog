"""An adapter for the Blue Gene/L log data available at: https://www.usenix.org/cfdr-data."""


import codecs
import datetime
import os

import click


class LogEntry:
    """A log entry."""

    code_to_key = {}

    def __init__(self, code, type, severity, timestamp):
        """Initialize a LogEntry.

        code -- [str] code of the log entry.
        type -- [str] type of the log entry.
        severity -- [str] severity of the log entry.
        timestamp -- [str] timestamp of the log entry.
        """
        if code not in LogEntry.code_to_key:
            LogEntry.code_to_key[code] = len(LogEntry.code_to_key) + 1
        self._code = code
        self._type = type
        self._severity = severity
        self._timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d-%H.%M.%S.%f")

    def __lt__(self, other):
        """Compare the timestamps of two log entries."""
        return self._timestamp < other._timestamp

    def key(self):
        """Return the identification number of this log entry."""
        return LogEntry.code_to_key[self._code]

    def is_anomaly(self):
        # """Return True if the severity of this log entry is FATAL, ERROR, SEVERE, or FAILURE."""
        # return self._severity in ["FATAL", "ERROR", "SEVERE", "FAILURE"]
        """Return True if the type of this log entry is defined."""
        return self._type != "-"

    def timestamp(self):
        """Return the timestamp of this log entry."""
        return self._timestamp


class LogData:
    """A list of log entries ordered by their timestamps."""

    def __init__(self, bgl2_path, logtemplatemap_path):
        """Parse log entries from files bgl2 and logTemplateMap.csv.

        bgl2_path -- [str] path to the bgl2 file.
        logtemplatemap_path -- [str] path to the logTemplateMap.csv file.
        """
        self._log_entries = []
        with codecs.open(bgl2_path, "r", encoding="utf-8", errors="ignore") as bgl2_file:
            with codecs.open(logtemplatemap_path, "r", encoding="utf-8", errors="ignore") as \
                    logtemplatemap_file:
                for (bgl2_line, logtemplatemap_line) in zip(bgl2_file, logtemplatemap_file):
                    code = logtemplatemap_line.strip()
                    bgl2_line_parts = bgl2_line.strip().split()
                    type = bgl2_line_parts[0]
                    timestamp = bgl2_line_parts[4]
                    for i in range(5, len(bgl2_line_parts)):
                        if bgl2_line_parts[i] in ["INFO", "FATAL", "WARNING", "SEVERE", "ERROR",
                                "FAILURE"]:
                            severity = bgl2_line_parts[i]
                            break
                    else:
                        assert False
                    self._log_entries.append(LogEntry(code, type, severity, timestamp))
        self._log_entries.sort()

    def fixedXY(self, window_size, window_step):
        """Return a tuple in which the first element is a list of event lists and the second is a
        list of target values. Each event list represents a fixed window built with the specified
        size and steps parameters.

        window_size -- [int] size of the window.
        window_size -- [int] step of the window.
        """
        X = []
        Y = []
        window_end_index = window_start_index = 0
        while window_end_index < len(self._log_entries):
            if window_end_index - window_start_index < window_size:
                window_end_index += 1
                continue
            # Build a window.
            x = []
            y = 0
            for i in range(window_start_index, window_end_index):
                x.append(self._log_entries[i])
                if self._log_entries[i].is_anomaly():
                    y = 1
            X.append(x)
            Y.append(y)
            # Slide the time window.
            new_window_start_index = window_start_index + window_step
            window_end_index = window_start_index = new_window_start_index
        return (X, Y)

    def timeXY(self, window_size, window_step):
        """Return a tuple in which the first element is a list of event lists and the second is a
        list of target values. Each event list represents a time window built with the specified
        size and step parameters.

        window_size -- [float] size of the time window in hours.
        window_step -- [float] step of the time window in hours.
        """
        X = []
        Y = []
        window_end_index = window_start_index = 0
        while window_end_index < len(self._log_entries):
            if (self._log_entries[window_end_index].timestamp() - \
                    self._log_entries[window_start_index].timestamp()).seconds <= window_size * 3600:
                window_end_index += 1
                continue
            # Build a window.
            x = []
            y = 0
            for i in range(window_start_index, window_end_index):
                x.append(self._log_entries[i])
                if self._log_entries[i].is_anomaly():
                    y = 1
            X.append(x)
            Y.append(y)
            # Slide the time window.
            new_window_start_index = window_start_index
            while new_window_start_index < len(self._log_entries) and \
                    (self._log_entries[new_window_start_index].timestamp() - \
                    self._log_entries[window_start_index].timestamp()).seconds < window_step * 3600:
                new_window_start_index += 1
            window_end_index = window_start_index = new_window_start_index
        return (X, Y)

    def statistics(self):
        """Return a string with statistics of this LogData."""
        intervals = []
        for i in range(1, len(self._log_entries)):
            intervals.append(
                (self._log_entries[i].timestamp() - self._log_entries[i - 1].timestamp()).microseconds
            )
        intervals.sort()
        return '\n'.join([
            "Number of log entries: %s" % len(self._log_entries),
            "Number of log entries labeled as anomalies: %s" %
                sum([1 if log_entry.is_anomaly() else 0 for log_entry in self._log_entries]),
            "Number of log message types: %s" % len(LogEntry.code_to_key),
            "Average time between consecutive log entries (in microseconds): %s" % (sum(intervals) / len(intervals)),
            "Median of times between consecutive log entries (in microseconds): %s" %
                intervals[int(len(intervals) / 2)]
        ])


@click.group()
def main():
    pass


@main.command()
@click.argument("inputdir_path", metavar="<inputdir_path>")
@click.argument("outputdir_path", metavar="<outputdir_path>")
@click.argument("window_size", metavar="<window_size>", type=int)
@click.argument("window_step", metavar="<window_step>", type=int)
def fixed(inputdir_path, outputdir_path, window_size, window_step):
    "Build log key sequences from files bgl2 and logTemplateMap.csv aggregated with fixed windows."
    log_data = LogData(
        os.path.join(inputdir_path, "bgl2"),
        os.path.join(inputdir_path, "logTemplateMap.csv")
    )
    print(log_data.statistics())
    X, Y = log_data.fixedXY(window_size, window_step)
    with open(os.path.join(outputdir_path, "normal"), 'w') as normal_file:
        with open(os.path.join(outputdir_path, "abnormal"), 'w') as abnormal_file:
            for (x, y) in zip(X ,Y):
                if y == 0:
                    normal_file.write(' '.join([str(log_entry.key()) for log_entry in x]) + '\n')
                else:
                    abnormal_file.write(' '.join([str(log_entry.key()) for log_entry in x]) + '\n')


@main.command()
@click.argument("inputdir_path", metavar="<inputdir_path>")
@click.argument("outputdir_path", metavar="<outputdir_path>")
@click.argument("window_size", metavar="<window_size>", type=float)
@click.argument("window_step", metavar="<window_step>", type=float)
def time(inputdir_path, outputdir_path, window_size, window_step):
    "Build log key sequences from files bgl2 and logTemplateMap.csv aggregated with time windows."
    log_data = LogData(
        os.path.join(inputdir_path, "bgl2"),
        os.path.join(inputdir_path, "logTemplateMap.csv")
    )
    print(log_data.statistics())
    X, Y = log_data.timeXY(window_size, window_step)
    with open(os.path.join(outputdir_path, "normal"), 'w') as normal_file:
        with open(os.path.join(outputdir_path, "abnormal"), 'w') as abnormal_file:
            for (x, y) in zip(X ,Y):
                if y == 0:
                    normal_file.write(' '.join([str(log_entry.key()) for log_entry in x]) + '\n')
                else:
                    abnormal_file.write(' '.join([str(log_entry.key()) for log_entry in x]) + '\n')


if __name__ == "__main__":
    main()
