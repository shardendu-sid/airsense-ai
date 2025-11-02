from app.auth.models import User
from app import create_app, db

def flask_runner():
    flask_app = create_app('dev')
    with flask_app.app_context():
        db.create_all()
        # Create admin user if it doesn't exist
        if not User.query.filter_by(user_name='admin').first():
            User.create_user(
                user='admin',
                email='admin@admin.com',
                password='ticktock'
            )

    # Disable debug/reloader so signals propagate correctly
    flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

if __name__ == '__main__':
    flask_runner()
