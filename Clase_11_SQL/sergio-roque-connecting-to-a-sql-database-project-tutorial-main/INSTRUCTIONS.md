# Connecting to an SQL database using Python

## Part 1: Install Dependencies

There is a file in this project called `./requirements.txt` which contains the list of Python libraries we will use in this project, such as Pandas, psycopg2, SQLAlchemy, etc. Run the command `pip install -r requirements.txt` to install all the libraries at once using the PIP (Package Installer for Python).

> NOTE: Ensure that the command does not produce any errors in the terminal.

## Part 2: Add Your Database Credentials

To connect your application to the database, you need to configure a  `.env`  file at the root of the project. This file securely stores sensitive credentials and configurations and should be ignored by Git to prevent exposing confidential information. At the root of the project, you’ll find a file called .env.example, which contains a reference format for the database credentials.

Create and open the  `.env`  file and add your database credentials.

If you are running the database locally:

    
    DB_HOST="localhost" 
    DB_PORT= 3306
    DB_USER="tu_usuario"
    DB_PASSWORD="tu_contraseña"
    DB_NAME="nombre_de_tu_base"
    

If your database is hosted on a cloud service, use the credentials provided by your provider (the values below are examples):

    
    DB_HOST="f565gmi022AD.cbetxkdyhwsb.us-east-1.rds.amazonaws.com"
    DB_PORT=3306  # Ajusta el puerto según el servicio
    DB_USER="hkietatgd83b4x0l"
    DB_PASSWORD="p0s2wasdado1cr02d12"
    DB_NAME="y9uflxvx2hsf11g3f"
    
## Part 3 : Create a Database Using PostgreSQL

Make sure you have installed the Postgres client for the terminal called `PSQL`. You can check if you have it by running the following command:

```bash
$ psql --version
```

> NOTE: If you get an error, try [following these instructions](https://www.timescale.com/blog/how-to-install-psql-on-mac-ubuntu-debian-windows/) to install `psql` on your computer.

Now, review the contents of the `./.env` file and follow the steps described below:

1. Postgres is a database server, you have to start the server before you can use it. In order to stat postgres in your computer run the following command: `sudo service postgresql start`
2. Create a new user to connect to your database: `psql -U postgres -c "CREATE USER gitpod;"`
    - psql -U postgres → Connect to PostgreSQL using the default admin user (postgres).
    - -c "CREATE USER <DB_USER>;" → Execute the SQL command to create a user named <DB_USER>.

        If in the .env file you have:

            DB_USER = 'my_user'
        The command you should run is:

            `$ psql -U postgres -c "CREATE USER my_user;"` 


3. Create a new database within the Postgres engine by customizing and executing the following command: `psql -U postgres -c "CREATE DATABASE <DB_NAME> OWNER <DB_USER>;"`

        If in your .env file you have:

                DB_NAME = 'my_database'
                DB_USER = 'my_user'

        Then the command would be:

            `$ psql -U postgres -c "CREATE DATABASE my_database OWNER my_user;"`
4. Connect to the Postgres engine to use your database, manipulate tables and data: `$ psql -h localhost -U <username> <db_name>`

> NOTE: Remember to check the `./.env` file information to get or set the `username` and `db_name`.

Once you are inside `PSQL` you will be able to create tables, make queries, insert, update, or delete data and much more!

## Part 4: Understand the structure of your project's template

Once you have completed the above steps, you will have created your SQL database, but there are no tables yet. Next, you need to connect to your empty database, and you will create some tables using Python via SQL scripts that you will find in the `./src/sql/` folder:

- `./src/sql/create.sql` with all the tables you need to create. Hands on the creation of those tables.
- `./src/sql/insert.sql` with all the table values that will be inserted into each table.
- `./src/sql/drop.sql` contains the SQL code to drop the tables, very useful to clean up your database.

Other important things to mention about the structure:

- Professional projects usually have a `./src` folder that contains all the **coding files** (the files that the developer will implement).
- `./assets` is irrelevant to you; it contains all the images we needed for this tutorial.
- The root of any professional project usually contains all the configuration files such as `.gitignore`, `requirements.txt`, etc. You will learn more about these files during the rest of the project.

## Part 5: Start programming

All your Python code should always be inside the `src` folder; this is also another good practice.

1. The `connect` function contains the code needed to connect to your Python database. If you look closely, you will see how it loads all the environment variables into a variable called `connection_string` and then calls the `create_engine` and `connect` functions.

```py
def connect():
    global engine
    try:
        connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        print("Starting the connection...")
        engine = create_engine(connection_string, isolation_level="AUTOCOMMIT")
        engine.connect()
        print("Connected successfully!")
        return engine
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
```

## 📝 Instructions

1. We will work on the file inside the src folder called `./src/app.py.`

> NOTE: This file will contain most of your application code, as well as the database connection and database queries.

2. Create a PostgreSQL database as instructed in the guide.

3. In your `app.py`, program the connection to your database.

> NOTE: You can try to reuse the `connect` Python function provided above.

4. Create the tables indicated in `./src/sql/create.sql`.

5. Insert the data specified in `./src/sql/insert.sql` into the tables created in the previous point.

6. Use Pandas to print one of the tables as a DataFrame using the `read_sql` function of this library.

## Additional tips

- Log into your database using `psql -h localhost -u postgres`
- Connect to a remote database using: `psql -h <REMOTE_HOST> -p <REMOTE_PORT> -U <DB_USER> <DB_NAME>`
- Deletes a database: `$ dropdb -h localhost -U <username> <db_name>`
- Create a DB in the render.com cloud and connect [following these commands](https://render.com/docs/databases#connecting-from-outside-render).

