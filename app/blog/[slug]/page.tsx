import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';

import { POSTS, formatDate } from '../posts';

export const generateStaticParams = async () =>
    POSTS.map((post) => ({ slug: post.slug }));

export const generateMetadata = async ({
    params,
}: {
    params: Promise<{ slug: string }>;
}): Promise<Metadata> => {
    const { slug } = await params;
    const post = POSTS.find((item) => item.slug === slug);
    if (!post) return {};

    return {
        title: `${post.title} - Vishal Krishna Kumar`,
        description: post.summary,
        authors: [{ name: 'Vishal Krishna Kumar' }],
        openGraph: {
            title: post.title,
            description: post.summary,
            type: 'article',
            publishedTime: post.date,
            authors: ['Vishal Krishna Kumar'],
            url: `https://www.vishalkrishna.dev/blog/${post.slug}`,
        },
    };
};

const Page = async ({ params }: { params: Promise<{ slug: string }> }) => {
    const { slug } = await params;
    const post = POSTS.find((item) => item.slug === slug);
    if (!post) return notFound();

    const { Content } = post;

    return (
        <article className="pt-10 pb-24">
            <div className="container max-w-[760px]">
                <Link
                    href="/blog"
                    className="mb-12 inline-flex gap-2 items-center group h-12 text-muted-foreground hover:text-primary transition-colors"
                >
                    <ArrowLeft className="group-hover:-translate-x-1 transition-transform duration-300" />
                    All posts
                </Link>

                <p className="text-sm text-muted-foreground mb-4">
                    {formatDate(post.date)} · {post.readingMinutes} min read ·
                    Vishal Krishna Kumar
                </p>
                <h1 className="font-anton text-4xl md:text-6xl leading-[1.05] mb-6">
                    {post.title}
                </h1>
                <p className="text-xl text-muted-foreground mb-12">
                    {post.summary}
                </p>

                <Content />
            </div>
        </article>
    );
};

export default Page;
