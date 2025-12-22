"""
Universal Data Connectors Demo

Shows how to connect to 13+ data sources with VizForge.
"""

import vizforge as vz


def demo_database_connectors():
    """Demo 1: Database Connectors"""
    print("=" * 60)
    print("DEMO 1: Database Connectors")
    print("=" * 60)

    # PostgreSQL
    print("\n📊 PostgreSQL Example:")
    print("""
    db = vz.connect('postgresql',
        host='localhost',
        database='mydb',
        username='user',
        password='pass'
    )
    df = db.query("SELECT * FROM users")
    chart = vz.bar(df, x='name', y='count')
    """)

    # MySQL
    print("\n📊 MySQL Example:")
    print("""
    db = vz.connect('mysql',
        host='localhost',
        database='mydb',
        username='user',
        password='pass'
    )
    df = db.read(table='sales')
    """)

    # SQLite
    print("\n📊 SQLite Example:")
    print("""
    db = vz.connect('sqlite', path='data.db')
    df = db.query("SELECT * FROM products")
    """)

    # MongoDB
    print("\n📊 MongoDB Example:")
    print("""
    mongo = vz.connect('mongodb',
        host='localhost',
        database='mydb'
    )
    df = mongo.read('users', {'age': {'$gt': 25}})
    """)

    print("\n✅ Demo complete!")


def demo_cloud_connectors():
    """Demo 2: Cloud Storage Connectors"""
    print("\n" + "=" * 60)
    print("DEMO 2: Cloud Storage Connectors")
    print("=" * 60)

    # AWS S3
    print("\n☁️ AWS S3 Example:")
    print("""
    s3 = vz.connect('s3',
        bucket='my-bucket',
        username='AWS_KEY',
        password='AWS_SECRET'
    )
    df = s3.read('data/sales.csv', file_type='csv')
    chart = vz.line(df, x='date', y='revenue')
    """)

    # Google Cloud Storage
    print("\n☁️ Google Cloud Storage Example:")
    print("""
    gcs = vz.connect('gcs',
        bucket='my-bucket',
        options={'project': 'my-project'}
    )
    df = gcs.read('data/users.parquet', file_type='parquet')
    """)

    # Azure Blob
    print("\n☁️ Azure Blob Storage Example:")
    print("""
    azure = vz.connect('azure',
        bucket='my-container',
        username='account_name',
        password='account_key'
    )
    df = azure.read('data/metrics.json', file_type='json')
    """)

    print("\n✅ Demo complete!")


def demo_api_connectors():
    """Demo 3: API Connectors"""
    print("\n" + "=" * 60)
    print("DEMO 3: API Connectors")
    print("=" * 60)

    # REST API
    print("\n🌐 REST API Example:")
    print("""
    api = vz.connect('rest',
        url='https://api.example.com',
        api_key='YOUR_API_KEY'
    )
    df = api.read('/users')
    chart = vz.scatter(df, x='age', y='score')
    """)

    # GraphQL
    print("\n🌐 GraphQL API Example:")
    print("""
    graphql = vz.connect('graphql',
        url='https://api.example.com/graphql',
        api_key='YOUR_API_KEY'
    )

    query = '''
        query {
            users {
                name
                email
                age
            }
        }
    '''

    df = graphql.read(query)
    """)

    print("\n✅ Demo complete!")


def demo_file_web_connectors():
    """Demo 4: File & Web Connectors"""
    print("\n" + "=" * 60)
    print("DEMO 4: File & Web Connectors")
    print("=" * 60)

    # Excel
    print("\n📁 Excel Example:")
    print("""
    excel = vz.connect('excel', path='data.xlsx')
    df = excel.read(sheet_name='Sales')
    """)

    # Parquet
    print("\n📁 Parquet Example:")
    print("""
    parquet = vz.connect('parquet', path='data.parquet')
    df = parquet.read()
    """)

    # HDF5
    print("\n📁 HDF5 Example:")
    print("""
    hdf5 = vz.connect('hdf5', path='data.h5')
    df = hdf5.read(key='dataset1')
    """)

    # HTML Table
    print("\n🌐 HTML Table Example:")
    print("""
    html = vz.connect('html', url='https://example.com/data.html')
    df = html.read(table_index=0)
    """)

    # Web Scraper
    print("\n🌐 Web Scraper Example:")
    print("""
    web = vz.connect('web', url='https://example.com')
    df = web.read(selector='.data-table tr')
    """)

    print("\n✅ Demo complete!")


def demo_list_connectors():
    """Demo 5: List Available Connectors"""
    print("\n" + "=" * 60)
    print("DEMO 5: Available Connectors")
    print("=" * 60)

    connectors = vz.list_connectors()

    for category, types in connectors.items():
        print(f"\n{category}:")
        for connector_type in types:
            print(f"  • {connector_type}")

    print("\n✅ Total: 13+ connectors available!")


def main():
    """Run all connector demos."""
    print("\n" + "=" * 70)
    print("🔌 VizForge Universal Data Connectors - Comprehensive Demo")
    print("=" * 70)

    print("\nConnect to 13+ data sources with ONE unified interface!")
    print("\nSupported Sources:")
    print("  • Databases: PostgreSQL, MySQL, SQLite, MongoDB")
    print("  • Cloud: AWS S3, Google Cloud, Azure Blob")
    print("  • APIs: REST, GraphQL")
    print("  • Files: Excel, Parquet, HDF5")
    print("  • Web: HTML Tables, Web Scraping")

    # Run demos
    demo_database_connectors()
    demo_cloud_connectors()
    demo_api_connectors()
    demo_file_web_connectors()
    demo_list_connectors()

    print("\n" + "=" * 70)
    print("✅ ALL CONNECTOR DEMOS COMPLETE!")
    print("=" * 70)

    print("\n🚀 Quick Start:")
    print("""
    import vizforge as vz

    # Connect to any source
    db = vz.connect('postgresql', ...)
    df = db.query("SELECT * FROM table")

    # Create chart
    chart = vz.line(df, x='date', y='value')
    chart.show()
    """)


if __name__ == '__main__':
    main()
