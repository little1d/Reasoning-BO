from pymilvus import connections
from neo4j import GraphDatabase
import pytest


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_milvus_connection():
    # host 填 docker 容器所在机器 ip
    connections.connect(alias="default", host='10.140.52.87', port='19530')

    connection_list = connections.list_connections()
    assert len(connection_list) == 1
    assert connection_list[0][0] == "default"
    assert connection_list[0][1] is not None


def test_neo4j_connection():
    URI = "bolt://10.140.52.87:7687"
    AUTH = ("neo4j", "123456789")

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
