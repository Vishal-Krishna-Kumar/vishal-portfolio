import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';

import { POSTS, formatDate } from './posts';

export const metadata: Metadata = {
    title: 'Blog - Vishal Krishna Kumar',
    description:
        'Notes on retrieval, RAG evaluation and performance, with measured results and open-source code.',
};

const BlogIndex = () => (
    <section className="pt-24 pb-24">
        <div className="container max-w-[820px]">
            <p className="text-muted-foreground uppercase tracking-widest text-sm mb-4">
                Blog
            </p>
            <h1 className="font-anton text-5xl md:text-7xl leading-none mb-6">
                Things I measured
            </h1>
            <p className="text-lg text-muted-foreground mb-14 max-w-[620px]">
                Short write-ups on retrieval, RAG evaluation and performance.
                Every number comes from code you can run.
            </p>

            <ul className="divide-y divide-border border-y border-border">
                {POSTS.map((post) => (
                    <li key={post.slug}>
                        <Link
                            href={`/blog/${post.slug}`}
                            className="group block py-8"
                        >
                            <p className="text-sm text-muted-foreground mb-2">
                                {formatDate(post.date)} · {post.readingMinutes}{' '}
                                min read
                            </p>
                            <h2 className="font-anton text-2xl md:text-3xl leading-tight mb-3 group-hover:text-primary transition-colors flex items-start gap-2">
                                {post.title}
                                <ArrowUpRight className="shrink-0 mt-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </h2>
                            <p className="text-lg text-foreground/80 mb-4">
                                {post.summary}
                            </p>
                            <div className="flex flex-wrap gap-2">
                                {post.tags.map((tag) => (
                                    <span
                                        key={tag}
                                        className="text-xs uppercase tracking-wide border border-border rounded-full px-3 py-1 text-muted-foreground"
                                    >
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </Link>
                    </li>
                ))}
            </ul>
        </div>
    </section>
);

export default BlogIndex;
