import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Thread, Lock
from time import sleep


class teleLogger:
    def __init__(self, filename="telemetry.parquet", batch_size=50):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        self.lock = Lock()
        self.running = True

        # Fixed column order (this is the schema contract)
        self.columns = ["t", "q0", "q1", "q2", "q3", "w0", "w1", "w2", "fault"]

        # ✅ IMPORTANT FIX: define a typed schema explicitly (avoid Arrow "null" types)
        self.schema = pa.schema([
            ("t", pa.float64()),
            ("q0", pa.float64()), ("q1", pa.float64()), ("q2", pa.float64()), ("q3", pa.float64()),
            ("w0", pa.float64()), ("w1", pa.float64()), ("w2", pa.float64()),
            ("fault", pa.int64())
        ])

        # Create the Parquet file if it doesn't exist (with the correct schema)
        if not os.path.exists(self.filename):
            empty_table = pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in self.schema],
                schema=self.schema
            )
            pq.write_table(empty_table, self.filename)

        # Open writer with the fixed schema
        self.writer = pq.ParquetWriter(self.filename, self.schema, use_dictionary=True)

        # Start background thread to flush periodically
        self.thread = Thread(target=self._background_flush, daemon=True)
        self.thread.start()

    def log(self, t, q, w, fault=0):
        """
        Add a telemetry entry to the buffer.
        t: float timestamp
        q: quaternion [q0,q1,q2,q3]
        w: angular velocity [w0,w1,w2]
        fault: int (0 or 1)
        """
        entry = {"t": float(t)}
        for i, val in enumerate(q):
            entry[f"q{i}"] = float(val)
        for i, val in enumerate(w):
            entry[f"w{i}"] = float(val)
        entry["fault"] = int(fault)

        flush_now = False
        with self.lock:
            self.buffer.append(entry)
            flush_now = len(self.buffer) >= self.batch_size

        if flush_now:
            self._flush()

    def _background_flush(self):
        """
        Background thread: flush buffer once per second.
        """
        while self.running:
            sleep(1)
            self._flush()

    def _flush(self):
        """
        Write the buffer to Parquet using the fixed schema, then clear the buffer.
        """
        with self.lock:
            if not self.buffer:
                return

            # Force the correct column order every time
            df = pd.DataFrame(self.buffer, columns=self.columns)

            # Convert using the fixed schema (prevents type/column drift)
            table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)

            self.writer.write_table(table)
            self.buffer = []

    def stop(self):
        """
        Stop background thread, flush remaining data, close the writer.
        """
        self.running = False
        self.thread.join()
        self._flush()
        self.writer.close()
