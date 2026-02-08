/**
 * TypeScript type definitions for the sample web application.
 */

export interface User {
    id: string;
    username: string;
    email: string;
    role: UserRole;
    isActive: boolean;
    profile: UserProfile;
    createdAt: string;
}

export enum UserRole {
    Guest = "guest",
    User = "user",
    Moderator = "moderator",
    Admin = "admin",
}

export interface UserProfile {
    bio: string;
    avatarUrl: string;
    location: string;
    website: string;
}

export interface Post {
    id: string;
    title: string;
    content: string;
    authorId: string;
    status: PostStatus;
    tags: string[];
    viewCount: number;
    likeCount: number;
    createdAt: string;
    updatedAt: string;
}

export enum PostStatus {
    Draft = "draft",
    Published = "published",
    Archived = "archived",
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
}

export interface ApiError {
    error: string;
    status: number;
    details?: Record<string, string>;
}

export type ApiResponse<T> = T | ApiError;

export interface SearchQuery {
    query: string;
    tag?: string;
    page?: number;
    pageSize?: number;
}

export interface AuthToken {
    token: string;
    expiresAt: string;
}
