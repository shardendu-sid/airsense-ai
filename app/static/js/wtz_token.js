function fetchDashboardData() {
    console.log('Fetching dashboard data...');
    
    const token = localStorage.getItem('access_token');
    console.log('Token:', token);

    if (!token) {
        console.error('Access token not found');
        window.location.href = '/login.html'; // Redirect to login if token is not available
        return;
    }

    fetch('/dashboard', {
        headers: {
            'Authorization': 'Bearer ' + token
        }
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else if (response.status === 401) {
            console.error('Unauthorized: Redirecting to login page');
            window.location.href = '/login.html';
        } else {
            throw new Error('Failed to fetch dashboard data');
        }
    })
    .then(data => {
        console.log('Dashboard data:', data);
        // Handle dashboard data
    })
    .catch(error => {
        console.error('Error fetching dashboard data:', error);
    });
}

console.log('Calling fetchDashboardData...');
fetchDashboardData();