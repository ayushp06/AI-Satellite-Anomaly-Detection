import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Thread, Lock
from time import sleep


class teleLogger:
    def __init__(self, filename="telemetry.parquet", batch_size=50):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []      # in-memory storage
        self.lock = Lock()    # prevents race conditions
        self.running = True   # controls the background thread

        self.columns = ["t", "q0", "q1", "q2", "q3", "w0", "w1", "w2"]

        # If file doesn't exist, create it with proper schema
        if not os.path.exists(self.filename):
            empty_df = pd.DataFrame(columns=self.columns)
            table = pa.Table.from_pandas(empty_df)
            pq.write_table(table, self.filename)

        # Initialize ParquetWriter in append mode
        self.writer = pq.ParquetWriter(self.filename, table.schema, use_dictionary=True)

        # Start background thread
        self.thread = Thread(target=self._background_flush)
        self.thread.start()


    def log(self, t, q, w):
        """
        Add a new telemetry entry to the buffer.
        t: timestamp
        q: quaternion [q0,q1,q2,q3]
        w: angular velocity [w0,w1,w2]
        """
        # Flatten the telemetry entry
        entry = {"t": round(float(t), 3)}
        for i, val in enumerate(q):
            entry[f"q{i}"] = float(val)
        for i, val in enumerate(w):
            entry[f"w{i}"] = float(val)

        # Add entry safely
        with self.lock:
            self.buffer.append(entry)

        # Optional: flush immediately if buffer reaches batch size
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _background_flush(self):
        """
        Background thread that periodically flushes buffer to Parquet.
        """
        while self.running:
            sleep(1)  # check buffer every second
            self._flush()

    def _flush(self):
        """
        Write buffer to Parquet and clear it.
        """
        with self.lock:
            if not self.buffer:
                return

            # Convert buffer to PyArrow Table
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df)

            # Append the batch to the Parquet file
            self.writer.write_table(table)

            # Clear buffer after flushing
            self.buffer = []
    
    def stop(self):
        """
        Stop the background thread and flush remaining telemetry.
        """
        self.running = False          # stop background thread
        self.thread.join()            # wait for it to finish
        self._flush()                 # flush any remaining data
        self.writer.close()           # close ParquetWriter safely

