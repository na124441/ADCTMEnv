"""
Main entry point for the ADCTM (Autonomous Data Centre Thermal Management) application.
This script simply imports the FastAPI application instance and the main execution 
function from the `server.app` module, acting as a convenient top-level wrapper.
"""
from server.app import app, main


if __name__ == "__main__":
    # If the script is executed directly (not imported), 
    # invoke the main server logic to start the application.
    main()
