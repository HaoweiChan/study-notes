import { Badge } from "@/components/ui/badge";
import { CATEGORY_COLORS, FALLBACK_COLORS } from "@/types";

interface CategoryBadgeProps {
  category: string;
  className?: string;
}

export function CategoryBadge({ category, className }: CategoryBadgeProps) {
  const colorClass = CATEGORY_COLORS[category] || FALLBACK_COLORS[category.length % FALLBACK_COLORS.length];
  
  return (
    <Badge className={`${colorClass} hover:${colorClass} text-white ${className}`}>
      {category}
    </Badge>
  );
}


