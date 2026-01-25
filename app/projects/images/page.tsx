import { notFound } from 'next/navigation';
import { PROJECTS } from '@/lib/data';
import { Metadata } from 'next';

export const generateStaticParams = async () => {
  // Only generate params for images in all projects
  return PROJECTS.flatMap((project) =>
    project.images.map((img) => ({ image: img.src.split('/').pop() }))
  );
};

export const generateMetadata = async ({ params }: { params: Promise<{ image: string }> }) => {
  const { image } = await params;
  let found;
  PROJECTS.forEach((project) => {
    if (!found) {
      found = project.images.find((img) => img.src.endsWith(image));
    }
  });
  return {
    title: found?.title || image,
    description: found?.description,
  } as Metadata;
};

const Page = async ({ params }: { params: Promise<{ image: string }> }) => {
  const { image } = await params;
  let found, projectTitle;
  PROJECTS.forEach((project) => {
    if (!found) {
      found = project.images.find((img) => img.src.endsWith(image));
      if (found) projectTitle = project.title;
    }
  });
  if (!found) return notFound();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-black/90 p-4">
      <div className="max-w-4xl w-full">
        <img
          src={found.src}
          alt={found.title || projectTitle || image}
          className="w-full rounded-lg shadow-lg mb-4"
        />
        {(found.title || found.description) && (
          <div className="bg-black/80 text-white p-6 rounded-lg shadow-lg">
            {found.title && <div className="font-bold text-2xl mb-2">{found.title}</div>}
            {found.description && <div className="text-lg opacity-90">{found.description}</div>}
            {projectTitle && <div className="mt-2 text-sm opacity-60">From project: {projectTitle}</div>}
          </div>
        )}
      </div>
    </div>
  );
};

export default Page;
