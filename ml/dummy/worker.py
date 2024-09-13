import os
import pickle

import httpx
import pika
from model import echo_model


def get_rmq(func):

    def wrapper(*args, **kwargs):
        rabbitmq_host = os.getenv('RABBITMQ_HOST')
        rabbitmq_port = os.getenv('RABBITMQ_PORT')
        rabbitmq_user = os.getenv('RABBITMQ_USER')
        rabbitmq_password = os.getenv('RABBITMQ_PASSWORD')
        params = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=rabbitmq_port,
            virtual_host='/',
            credentials=pika.PlainCredentials(
                username=rabbitmq_user,
                password=rabbitmq_password
            ),
            heartbeat=30,
            blocked_connection_timeout=2
        )
        connection = pika.BlockingConnection(params)
        result = func(connection, *args, **kwargs)
        connection.close()
        return result

    return wrapper


@get_rmq
def run(rmq, task_handler):

    def callback(ch, method, properties, body):
        try:
            task_handler(body)
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel = rmq.channel()
    channel.queue_declare(queue='dummy', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue='dummy',
        on_message_callback=callback,
        auto_ack=False  # Автоматическое подтверждение обработки сообщений
    )
    channel.start_consuming()


def process_task(task):
    task = pickle.loads(task)
    result = echo_model.run(task["input"])
    httpx.patch(task["callback_ep"], json={"output": result})


if __name__ == '__main__':
    run(process_task)

