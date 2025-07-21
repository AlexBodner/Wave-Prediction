import sqlite3
from collections import defaultdict

def connect(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    return conn, c

def close(conn):
    conn.close()

def get_all_topics_and_messages(cursor):
    # Get topic names and IDs
    cursor.execute("SELECT id, name FROM topics")
    topics = {row[0]: row[1] for row in cursor.fetchall()}

    # Get message sizes per topic
    size_per_topic = defaultdict(int)
    cursor.execute("SELECT topic_id, data FROM messages")
    for topic_id, data in cursor.fetchall():
        size_per_topic[topics[topic_id]] += len(data)

    return size_per_topic

if __name__ == "__main__":
    bag_file = 'datasets/primera_captura_13_03/rosbag2_2025_03_13-15_57_34_0.db3'

    # Connect to the database
    conn, c = connect(bag_file)

    # Calculate size per topic
    size_per_topic = get_all_topics_and_messages(c)

    # Print results
    total_size = sum(size_per_topic.values())
    print(f"Total size: {total_size / 1024 / 1024:.2f} MiB")
    for topic, size in size_per_topic.items():
        print(f"Topic: {topic} | Size: {size / 1024 / 1024:.2f} MiB ({size} bytes)")

    # Close connection
    close(conn)