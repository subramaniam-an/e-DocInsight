"""
Service for managing SQLite database operations.
"""
import sqlite3

class DatabaseService:
    """Service for managing SQLite database operations."""
    
    def __init__(self, db_path: str = "file_metadata.db"):
        """Initialize the database service."""
        self.db_path = db_path
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure the database and tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create file_metadata table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def clear_database(self):
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                # Delete all data from each table
                for table in tables:
                    table_name = table[0]
                    if table_name != 'sqlite_sequence':  # Skip sqlite_sequence table
                        cursor.execute(f"DELETE FROM {table_name}")
                        print(f"Cleared data from table: {table_name}")
                
                # Reset auto-increment counters
                cursor.execute("DELETE FROM sqlite_sequence")
                
                conn.commit()
                print("All tables cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing database: {str(e)}")
            return False 