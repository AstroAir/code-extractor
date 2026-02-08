/**
 * Frontend application entry point.
 *
 * Handles client-side routing, API calls, and UI rendering
 * for the sample web application.
 */

const API_BASE = "/api/v1";

class ApiClient {
    constructor(baseUrl = API_BASE) {
        this.baseUrl = baseUrl;
        this.token = null;
    }

    setToken(token) {
        this.token = token;
    }

    async request(method, path, data = null) {
        const headers = {
            "Content-Type": "application/json",
        };

        if (this.token) {
            headers["Authorization"] = `Bearer ${this.token}`;
        }

        const options = { method, headers };
        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.baseUrl}${path}`, options);
        return response.json();
    }

    async login(username, password) {
        const result = await this.request("POST", "/auth/login", {
            username,
            password,
        });
        if (result.token) {
            this.setToken(result.token);
        }
        return result;
    }

    async getUsers() {
        return this.request("GET", "/users");
    }

    async createPost(title, content, tags = []) {
        return this.request("POST", "/posts", { title, content, tags });
    }

    async searchPosts(query) {
        return this.request("GET", `/posts/search?q=${encodeURIComponent(query)}`);
    }
}

// TODO: Add WebSocket support for real-time updates
function initializeApp() {
    const client = new ApiClient();
    console.log("Application initialized");
    return client;
}

// Event handler for DOM content loaded
document.addEventListener("DOMContentLoaded", () => {
    const app = initializeApp();
    console.log("DOM ready, app:", app);
});

export { ApiClient, initializeApp };
